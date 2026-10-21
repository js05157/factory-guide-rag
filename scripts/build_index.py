"""프로젝트 루트에서 실행: python scripts/build_index.py"""

import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def make_records(alarms, manuals, splitter):
    # 문서 ID로 상세 설명을 바로 찾기 위한 딕셔너리
    manuals_by_id = {doc["doc_id"]: doc for doc in manuals["documents"]}
    alarm_records = []
    manual_records = []

    for alarm in alarms["items"]:
        item_id = alarm["item_id"]

        # 알람 하나를 검색 자료 하나로 구성
        alarm_records.append({
            "id": item_id,
            "text": alarm["search_text"],
            "metadata": {
                "item_id": item_id,
                "alarm_code": alarm["alarm_code"],
                "detail_code": alarm["detail_code"],
                "title": alarm["detail_name_en"],
                "source_type": "alarm_metadata",
                "source_url": alarm["source_url"],
                "pdf_page": alarm["source_pdf_page"],
                "manual_doc_ids_json": json.dumps(alarm["manual_doc_ids"]),
            },
        })

        # 이 알람에 연결된 상세 설명을 페이지별로 청킹
        for doc_id in alarm["manual_doc_ids"]:
            doc = manuals_by_id[doc_id]
            for page in doc["pages"]:
                page_number = page["pdf_page"]
                chunks = splitter.split_text(page["text"])

                for chunk_index, chunk in enumerate(chunks):
                    manual_records.append({
                        "id": f"{item_id}:{doc_id}:p{page_number}:c{chunk_index}",
                        "text": f"{doc['title']}\nPDF page: {page_number}\n\n{chunk}",
                        "metadata": {
                            "item_id": item_id,
                            "alarm_code": alarm["alarm_code"],
                            "detail_code": alarm["detail_code"],
                            "doc_id": doc_id,
                            "title": doc["title"],
                            "source_type": "pdf",
                            "source_url": doc["source_url"],
                            "pdf_page": page_number,
                            "chunk_index": chunk_index,
                            "referenced_doc_ids_json": json.dumps(doc["referenced_doc_ids"]),
                        },
                    })

    return alarm_records, manual_records


def embed_and_save(collection, records, api, model):
    # 한 번에 64개씩 임베딩하고 저장
    batch_size = 64
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        texts = [record["text"] for record in batch]

        response = api.embeddings.create(
            model=model, input=texts, encoding_format="float"
        )
        embeddings = [
            row.embedding for row in sorted(response.data, key=lambda row: row.index)
        ]

        collection.upsert(
            ids=[record["id"] for record in batch],
            documents=texts,
            metadatas=[record["metadata"] for record in batch],
            embeddings=embeddings,
        )
        print(f"{collection.name}: {start + len(batch)}/{len(records)}건 저장", flush=True)


def main():
    load_dotenv(Path.cwd() / ".env")

    alarms = load_json("data/processed/alarms_linked.json")
    manuals = load_json("data/processed/manual_docs.json")

    api = OpenAI(
        api_key=os.environ["GMS_KEY"],
        base_url=os.getenv("GMS_BASE_URL", "https://gms.ssafy.io/gmsapi/api.openai.com/v1"),
    )
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    )
    alarm_records, manual_records = make_records(alarms, manuals, splitter)

    db = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "data/chroma"))
    for name, records in [("alarm_meta", alarm_records), ("manual_docs", manual_records)]:
        collection = db.get_or_create_collection(
            name=name,
            embedding_function=None,
            metadata={"embedding_model": model},
            configuration={"hnsw": {"space": "cosine"}},
        )
        embed_and_save(collection, records, api, model)
        print(f"{name} 저장 완료: {collection.count()}건")


if __name__ == "__main__":
    main()
