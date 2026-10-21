"""MR-J4 공식 매뉴얼의 알람 목록을 JSON으로 추출하는 첫 실습.

프로젝트 루트에서 실행:
    python -m pip install pdfplumber
    python scripts/extract_alarms.py

기본 입력: data/raw/mr_j4_troubleshooting.pdf
기본 출력: data/processed/alarms_sample.json (10건)
이 스크립트는 MR-J4 영문 Troubleshooting 매뉴얼의 표 구조 전용입니다.
적용 모델과 원인/조치는 상세 설명 본문과 대조한 뒤 추가해야 합니다.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import pdfplumber


SOURCE_URL = (
    "https://dl.mitsubishielectric.com/dl/fa/document/manual/servo/"
    "sh030109eng/sh030109engs.pdf"
)
DETAIL_CODE = re.compile(r"[0-9A-F]{2}\.[0-9A-F]")


def clean(value):
    """셀 안의 줄바꿈과 여러 공백을 하나의 공백으로 정리합니다."""
    return " ".join((value or "").split())


def extract_alarms(pdf_path, limit):
    alarms = []
    seen = set()
    names_by_code = {}
    list_started = False

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = clean(page.extract_text())

            # 목차에는 두 제목이 함께 있으므로 건너뜁니다.
            is_alarm_list = "1.2 Alarm list" in text
            is_warning_list = "1.3 Warning list" in text
            if not list_started:
                if not is_alarm_list or is_warning_list:
                    continue
                list_started = True
            elif is_warning_list:
                break

            # 앞 4개 열: 상위코드 / 상위명 / 상세코드 / 상세명
            for table in page.extract_tables():
                for row in table:
                    if len(row) < 4:
                        continue
                    detail_code = clean(row[2])
                    if not DETAIL_CODE.fullmatch(detail_code):
                        continue
                    if detail_code in seen:
                        continue

                    alarm_code = detail_code.split(".")[0]
                    if clean(row[1]):
                        names_by_code[alarm_code] = clean(row[1])
                    name = names_by_code.get(alarm_code)
                    detail_name = clean(row[3])
                    if not name or not detail_name:
                        raise ValueError(
                            f"PDF {page_number}쪽: {detail_code}의 이름을 "
                            "읽지 못했습니다. 원문 표를 확인하세요."
                        )

                    alarms.append({
                        "item_id": f"mitsubishi_mr_j4_{detail_code.replace('.', '_')}",
                        "manufacturer": "Mitsubishi Electric",
                        "product_family": "MELSERVO-J4",
                        "alarm_code": alarm_code,
                        "detail_code": detail_code,
                        "alarm_name_en": name,
                        "detail_name_en": detail_name,
                        "applicable_models": None,
                        "review_status": "needs_manual_review",
                        "source_url": SOURCE_URL,
                        "source_pdf_page": page_number,
                        "search_text": (
                            f"Mitsubishi Electric MELSERVO-J4 | "
                            f"AL.{detail_code} | {name} | {detail_name}"
                        ),
                    })
                    seen.add(detail_code)
                    if limit and len(alarms) >= limit:
                        return alarms

    if not alarms:
        raise ValueError("알람을 추출하지 못했습니다. PDF 파일과 표 구조를 확인하세요.")
    if limit and len(alarms) < limit:
        raise ValueError(f"요청한 {limit}건 중 {len(alarms)}건만 추출했습니다.")
    return alarms


def main():
    parser = argparse.ArgumentParser(description="MR-J4 알람 목록 JSON 추출")
    parser.add_argument("--pdf", type=Path,
                        default=Path("data/raw/mr_j4_troubleshooting.pdf"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/alarms_sample.json"))
    parser.add_argument("--limit", type=int, default=10,
                        help="추출 건수. 0은 알람 목록 전체입니다.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit은 0 이상이어야 합니다.")
    if not args.pdf.is_file():
        parser.error(f"PDF를 찾지 못했습니다: {args.pdf}")

    try:
        alarms = extract_alarms(args.pdf, args.limit)
    except ValueError as error:
        parser.exit(1, f"추출 실패: {error}\n")

    result = {
        "document_title": "MR-J4 Servo Amplifier Instruction Manual (Troubleshooting)",
        "document_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "source_url": SOURCE_URL,
        "page_numbering": "1-based PDF page index, not printed page number",
        "count": len(alarms),
        "note": "자동 추출 초안. 적용 모델과 원인/조치는 원문 대조 후 추가하세요.",
        "items": alarms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"추출 완료: {len(alarms)}건")
    print(f"저장 위치: {args.output}")
    for item in alarms[:3]:
        print(f"  {item['detail_code']} | {item['alarm_name_en']} | "
              f"{item['detail_name_en']}")


if __name__ == "__main__":
    main()
