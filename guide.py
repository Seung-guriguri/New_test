"""In-app guidance: the collapsible guide box above the tabs, one-look tab intros and plain-language result sentences.
Guide texts live in guides/*.md so plant engineers can edit or add equipment without touching code."""
from pathlib import Path

import streamlit as st

GUIDES = Path(__file__).parent / "guides"


def _guides():
    out = {}
    for f in sorted(GUIDES.glob("*.md")):
        text = f.read_text(encoding="utf-8").strip()
        title, _, body = text.partition("\n")
        out[title.lstrip("# ").strip()] = body.strip()
    return out


def guide_box(open_=False):
    guides = _guides()
    if not guides:
        return
    with st.expander("📘 분석 가이드 — 무엇을 어떻게 분석할지 모르겠다면 펼치세요", expanded=open_):
        pick = st.segmented_control("무엇을 분석하시겠어요?", list(guides), default=list(guides)[0], key="guide_pick")
        st.markdown(guides.get(pick or list(guides)[0], ""))
        st.caption("가이드 내용은 프로그램 폴더의 `guides/*.md` 파일입니다. 우리 공정에 맞게 고치거나 파일을 추가하면 바로 반영됩니다.")


def tab_intro(when, pick, see):
    """Same three lines at the top of every main tab: when to use it, what to choose, what to look at."""
    st.caption(f"🧭 **언제** {when}  ·  **고를 것** {pick}  ·  **볼 것** {see}")


def checklist(name, title, items):
    """What to confirm before trusting a result. items: [(text, auto, why)] — auto True/False is the program's own
    check (shown with its reason), None means the engineer confirms it with a checkbox (kept while switching tabs, not
    saved in the settings file). Returns (done, total, report lines)."""
    from equipment_ui import flag  # equipment_ui imports this module
    import report
    done, lines = 0, []
    with st.expander(f"✔ {title}"):
        st.caption("✅/⚠️ 는 프로그램이 확인한 항목, 체크 상자는 직접 확인할 항목입니다. 모두 확인되기 전에는 결론을 보고하지 마세요.")
        for i, (text, auto, why) in enumerate(items):
            if auto is None:
                ok = flag(st, text, f"chk_{name}_{i}", help=why or None)
                mark = "☑" if ok else "☐"
            else:
                ok = bool(auto)
                st.markdown(f"{'✅' if ok else '⚠️'} {text} — _프로그램 확인: {why}_")
                mark = "✅" if ok else "⚠️"
            done += ok
            lines.append(f"- {mark} {text}" + (f" ({why})" if auto is not None and why else ""))
        st.markdown(f"**{done}/{len(items)} 확인됨**")
    report.put(f"확인 체크리스트 — {name}", [f"{done}/{len(items)} 확인됨", *lines])
    return done, len(items)
