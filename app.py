"""프로젝트 루트에서 실행: python -m streamlit run app.py"""

import json
import os
from pathlib import Path
from time import perf_counter

import streamlit as st
from dotenv import load_dotenv

from scripts.ask import (
    citation_numbers,
    load_rag,
    referenced_evidence,
    structured_evidence,
)


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="FactoryGuide", page_icon="🔧", layout="wide")
load_dotenv(ROOT / ".env")


@st.cache_resource(show_spinner="알람 목록과 매뉴얼을 불러오는 중입니다.")
def get_resources(root):
    return load_rag(root)


def page_url(metadata):
    return metadata["source_url"].split("#")[0] + f"#page={metadata['pdf_page']}"


def show_sources(answer, evidence):
    used = [n for n in citation_numbers(answer) if 1 <= n <= len(evidence)]
    if not used:
        st.caption("이 답변에는 문서 인용이 없습니다.")
    for number in used:
        row = evidence[number - 1]
        meta = row["metadata"]
        kind = "알람 목록" if meta["source_type"] == "alarm_metadata" else "매뉴얼 본문"
        with st.expander(f"[근거 {number}] {meta['title']} · PDF {meta['pdf_page']}쪽"):
            st.caption(kind)
            st.markdown(f"[원문 PDF 열기 · {meta['pdf_page']}쪽]({page_url(meta)})")
            st.text(row["text"])


def show_assistant(message):
    st.markdown(message["content"])
    st.caption(f"응답 시간 {message['seconds']:.1f}초")
    show_sources(message["content"], message["evidence"])
    with st.expander("검색어 확인"):
        st.write(" · ".join(message["keywords"]))


st.title("FactoryGuide")
st.caption("제조설비 알람을 선택하고, 공식 매뉴얼의 근거와 함께 답변을 확인하세요.")

if not os.getenv("GMS_KEY"):
    st.info("프로젝트의 .env 파일에 GMS_KEY를 설정한 후 실행해 주세요.")
    st.stop()

required_files = [
    ROOT / "data/processed/manual_docs.json",
    ROOT / "data/raw/mr_j4_troubleshooting.pdf",
]
missing = [str(path.relative_to(ROOT)) for path in required_files if not path.is_file()]
if missing:
    st.info("필요한 매뉴얼 파일이 없습니다: " + ", ".join(missing))
    st.stop()

db_path = ROOT / os.getenv("CHROMA_PATH", "data/chroma_large")
if not db_path.is_dir():
    st.info("색인이 없습니다. 기존 프로젝트에서 python scripts/build_index.py를 먼저 실행해 주세요.")
    st.stop()

resources = get_resources(str(ROOT))
if resources is None or not resources["alarms"]:
    st.info("알람 색인이 비어 있습니다. python scripts/build_index.py를 먼저 실행해 주세요.")
    st.stop()

alarms = resources["alarms"]
by_id = {row["id"]: row for row in alarms}
st.session_state.setdefault("histories", {})
st.session_state.setdefault("source_tables", {})

with st.sidebar:
    st.header("알람 찾기")
    keyword = st.text_input("코드·알람명 검색", placeholder="예: 72.4, encoder", key="alarm_search")
    parent_codes = sorted({row["metadata"]["alarm_code"] for row in alarms})
    parent = st.selectbox("상위 알람", ["전체"] + parent_codes, key="parent_filter")
    linked_only = st.checkbox("매뉴얼이 연결된 알람만", value=True)

    # 코드, 이름, 상위 알람, 매뉴얼 연결 여부로 후보를 좁힙니다.
    filtered = [
        row for row in alarms
        if (parent == "전체" or row["metadata"]["alarm_code"] == parent)
        and keyword.strip().casefold() in (
            row["metadata"]["detail_code"] + " " + row["metadata"]["title"]
        ).casefold()
        and (not linked_only or json.loads(row["metadata"]["manual_doc_ids_json"]))
    ]
    st.caption(f"검색 결과 {len(filtered)}건 / 전체 {len(alarms)}건")
    if not filtered:
        st.info("조건에 맞는 알람이 없습니다. 검색어 또는 필터를 변경해 주세요.")
        st.stop()

    ids = [row["id"] for row in filtered]
    default = next((i for i, row in enumerate(filtered) if row["metadata"]["detail_code"] == "72.4"), 0)
    item_id = st.selectbox(
        "상세 알람 선택", ids, index=default, key="selected_alarm",
        format_func=lambda value: f"AL.{by_id[value]['metadata']['detail_code']} · {by_id[value]['metadata']['title']}",
    )
    if st.button("이 알람의 대화 지우기", use_container_width=True):
        st.session_state.histories[item_id] = []

selected = by_id[item_id]["metadata"]
left, right = st.columns([3, 1])
with left:
    st.subheader(f"AL.{selected['detail_code']}")
    st.write(selected["title"].replace("non- signal", "non-signal"))
with right:
    st.caption("제조사 · 제품군")
    st.write("Mitsubishi Electric · MELSERVO-J4")

view = st.radio("화면", ["질의응답", "원문 점검표"], horizontal=True, label_visibility="collapsed")
st.caption("매뉴얼 기반 참고 정보입니다. 실제 점검은 현장 작업 절차를 따라 진행하세요.")

if view == "원문 점검표":
    st.caption("선택한 상세 알람과 명시적으로 참조된 알람의 영문 원문을 보여줍니다.")
    if st.button("원문 점검표 불러오기", type="primary"):
        with st.spinner("PDF 점검표를 읽는 중입니다."):
            rows = referenced_evidence(resources["sections"], selected["detail_code"])
            st.session_state.source_tables[item_id] = structured_evidence(rows, resources["pdf_path"])

    tables = st.session_state.source_tables.get(item_id, [])
    for row in tables:
        payload = json.loads(row["text"])
        meta = row["metadata"]
        st.markdown(f"### AL.{meta['detail_code']} · {payload['source_detail_name']}")
        st.markdown(f"[원문 PDF · {meta['pdf_page']}쪽]({page_url(meta)})")
        if payload["reference_codes"]:
            st.info("참조하는 상세 알람: " + ", ".join("AL." + code for code in payload["reference_codes"]))
        for i, procedure in enumerate(payload["procedures"], start=1):
            with st.container(border=True):
                st.markdown(f"**점검 {i} · 원인 후보 (원문)**")
                st.write(procedure["cause"])
                st.markdown("**확인 방법 (원문)**")
                st.write(procedure["check_method"])
                st.table([
                    {"확인 결과 (원문)": branch["check_result"], "조치 (원문)": branch["action"]}
                    for branch in procedure["branches"]
                ])
    if not tables:
        st.info("버튼을 누르면 AI 호출 없이 공식 매뉴얼의 점검표를 확인할 수 있습니다.")

else:
    history = st.session_state.histories.setdefault(item_id, [])
    st.caption("질문 예시: 이 알람의 의미와 점검 방법을 순서대로 알려줘.")
    if not history:
        st.info("선택한 알람에 대해 질문해 주세요. 대화 기록은 알람별로 보관됩니다.")
    for message in history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                show_assistant(message)

    question = st.chat_input("선택한 알람에 대해 질문하세요.", key=f"question_{item_id}")
    if question and question.strip():
        question = question.strip()
        # 현재 알람의 이전 대화 3쌍만 전달합니다. 현재 질문은 별도로 보냅니다.
        recent_history = [
            {"role": message["role"], "content": message["content"]}
            for message in history[-6:]
        ]
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            started = perf_counter()
            result = {}
            with st.status("관련 매뉴얼을 검색하고 있습니다.", expanded=True) as status:
                for update in resources["graph"].stream(
                    {"item_id": item_id, "question": question, "history": recent_history},
                    stream_mode="updates"
                ):
                    if "retrieve" in update:
                        result.update(update["retrieve"])
                        if not any(row["metadata"]["source_type"] == "pdf" for row in result["evidence"]):
                            st.warning("관련 매뉴얼 본문을 찾지 못했습니다. 답변의 확인 불가 안내를 확인해 주세요.")
                        status.write(f"검색한 근거 {len(result['evidence'])}건을 확인했습니다.")
                        status.update(label="검색한 근거로 답변을 작성하고 있습니다.")
                    if "generate" in update:
                        result.update(update["generate"])
                status.update(label="답변을 작성했습니다.", state="complete", expanded=False)
            message = {
                "role": "assistant", "content": result["answer"],
                "evidence": result["evidence"], "keywords": result["keywords"],
                "standalone_question": result["standalone_question"],
                "seconds": perf_counter() - started,
            }
            history.append(message)
            show_assistant(message)
