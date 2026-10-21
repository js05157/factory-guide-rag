"""MR-J4 매뉴얼의 알람별 상세 설명을 추출하고 알람 목록에 연결합니다.

프로젝트 루트에서 실행:
    python scripts/extract_manual.py

입력: data/raw/mr_j4_troubleshooting.pdf, data/processed/alarms.json
출력: data/processed/manual_docs.json, data/processed/alarms_linked.json

주의: 표를 읽기 순서의 원문 텍스트로 보존하는 추출 초안입니다.
원인/점검/결과/조치의 구조화와 적용 모델 검수는 별도로 진행해야 합니다.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import pdfplumber


HEADING = re.compile(r"^Alarm\s+No\.:\s*([0-9A-F]{2})\s+Name:\s*(.+)$")
DETAIL = re.compile(r"^([0-9A-F]{2}\.[0-9A-F])\s+", re.MULTILINE)
REFERENCE = re.compile(r"\[AL\.\s*([0-9A-F]{2}(?:\.[0-9A-F])?)\]")
SOURCE_URL = (
    "https://dl.mitsubishielectric.com/dl/fa/document/manual/servo/"
    "sh030109eng/sh030109engs.pdf"
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def extract_documents(pdf_path, fingerprint):
    grouped = {}
    active = False
    current_code = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").replace("\uf0d7", "•")
            compact = " ".join(text.split())
            if not active:
                # 목차에는 여러 절 제목이 함께 있으므로 제외합니다.
                if ("1.4 Remedies for alarms" not in compact
                        or "1.5 Remedies for warnings" in compact):
                    page.close()
                    continue
                active = True
                print(f"상세 설명 추출 시작: PDF {page_number}쪽", flush=True)
            elif "1.5 Remedies for warnings" in compact:
                page.close()
                break

            buffer = []

            def save_segment():
                if current_code and buffer:
                    body = "\n".join(buffer).strip()
                    if body:
                        grouped[current_code]["pages"].append({
                            "pdf_page": page_number,
                            "text": body,
                        })
                buffer.clear()

            for line in text.splitlines():
                match = HEADING.match(line.strip())
                if match:
                    save_segment()
                    current_code, name = match.groups()
                    grouped.setdefault(current_code, {
                        "alarm_code": current_code,
                        "alarm_name_en": name,
                        "pages": [],
                    })
                if current_code:
                    buffer.append(line)
            save_segment()
            page.close()
            if page_number % 10 == 0:
                print(f"PDF {page_number}쪽까지 처리 중...", flush=True)

    if not grouped:
        raise ValueError("알람 상세 설명을 찾지 못했습니다. PDF 버전과 구조를 확인하세요.")

    documents = []
    for code, group in grouped.items():
        parts = group["pages"]
        body = "\n\n".join(
            f"[PDF page {part['pdf_page']}]\n{part['text']}" for part in parts
        )
        details = sorted({c for c in DETAIL.findall(body)
                          if c.split(".")[0] == code})
        documents.append({
            "doc_id": f"mr_j4_{fingerprint[:12]}_alarm_{code}",
            "document_type": "manufacturer_manual",
            "source_type": "pdf",
            "manufacturer": "Mitsubishi Electric",
            "product_family": "MELSERVO-J4",
            "alarm_code": code,
            "title": f"AL.{code} - {group['alarm_name_en']}",
            "detail_codes": details,
            "referenced_codes": sorted(set(REFERENCE.findall(body))),
            "source_url": SOURCE_URL,
            "source_pdf_pages": sorted({part["pdf_page"] for part in parts}),
            "pages": parts,
            "text": body,
            "review_status": "needs_manual_review",
        })

    by_code = {doc["alarm_code"]: doc for doc in documents}
    for doc in documents:
        doc["referenced_doc_ids"] = sorted({
            by_code[code.split(".")[0]]["doc_id"]
            for code in doc["referenced_codes"]
            if code.split(".")[0] in by_code
            and code.split(".")[0] != doc["alarm_code"]
        })
    return documents


def main():
    parser = argparse.ArgumentParser(description="MR-J4 알람 상세 설명 추출")
    parser.add_argument("--pdf", type=Path,
                        default=Path("data/raw/mr_j4_troubleshooting.pdf"))
    parser.add_argument("--alarms", type=Path,
                        default=Path("data/processed/alarms.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed"))
    args = parser.parse_args()
    for path in (args.pdf, args.alarms):
        if not path.is_file():
            parser.error(f"파일을 찾지 못했습니다: {path}")

    metadata = read_json(args.alarms)
    fingerprint = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    expected = metadata.get("document_sha256")
    if expected and expected != fingerprint:
        parser.error("알람 목록과 PDF 버전이 다릅니다. 같은 PDF로 알람 목록을 다시 추출하세요.")
    try:
        documents = extract_documents(args.pdf, fingerprint)
    except ValueError as error:
        parser.exit(1, f"추출 실패: {error}\n")

    by_code = {doc["alarm_code"]: doc for doc in documents}
    linked_count = 0
    missing = []
    for item in metadata["items"]:
        code = item["alarm_code"]
        doc = by_code.get(code)
        # 상세코드가 원문에 실제로 등장하는지도 함께 확인합니다.
        if doc and item["detail_code"] in doc["detail_codes"]:
            item["manual_doc_ids"] = [doc["doc_id"]]
            item["manual_pdf_pages"] = doc["source_pdf_pages"]
            item["manual_link_status"] = "linked_needs_review"
            linked_count += 1
        else:
            item["manual_doc_ids"] = []
            item["manual_pdf_pages"] = []
            item["manual_link_status"] = "not_found"
            missing.append(item["detail_code"])

    write_json(args.output_dir / "manual_docs.json", {
        "document_sha256": fingerprint,
        "source_url": SOURCE_URL,
        "count": len(documents),
        "page_numbering": "1-based PDF page index, not printed page number",
        "note": "상위 알람별 원문 추출 초안. 표의 열 관계와 적용 모델은 검수 필요.",
        "documents": documents,
    })
    metadata["manual_link_summary"] = {
        "linked": linked_count, "not_found": len(missing),
        "missing_detail_codes": missing,
    }
    write_json(args.output_dir / "alarms_linked.json", metadata)
    print(f"상세 문서: {len(documents)}개")
    print(f"알람 연결: {linked_count}/{len(metadata['items'])}건")
    print(f"연결하지 못한 알람: {len(missing)}건")
    print(f"저장 위치: {args.output_dir}")
    if missing:
        print("원문 확인 필요: " + ", ".join(missing))


if __name__ == "__main__":
    main()
