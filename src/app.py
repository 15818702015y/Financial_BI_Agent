# src/app.py
import sys
import os
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from langchain_core.messages import HumanMessage
from src.graph import build_main_graph
from src.i18n import t

st.set_page_config(
    page_title="Financial BI Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_graph():
    """缓存 LangGraph 图，避免每次 rerun 都重建"""
    return build_main_graph()


# ========== 内联 SVG 图标库 ==========
ICONS = {
    "trend": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="{s}" height="{s}"><polyline points="3 17 9 11 13 15 21 7"/><polyline points="14 7 21 7 21 14"/></svg>',
    "clock": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="{s}" height="{s}"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "message-circle": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="{s}" height="{s}"><path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/></svg>',
    "sparkles": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="{s}" height="{s}"><path d="M12 3l1.9 4.4L18 9l-4.1 1.6L12 15l-1.9-4.4L6 9l4.1-1.6zM5 16l.9 2.1L8 19l-2.1.9L5 22l-.9-2.1L2 19l2.1-.9zM19 14l.9 2.1L22 17l-2.1.9L19 20l-.9-2.1L16 17l2.1-.9z"/></svg>',
    "bar-chart-3": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="{s}" height="{s}"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>',
    "search": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="{s}" height="{s}"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "file-text": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="{s}" height="{s}"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
}


def icon(name: str, size: int = 16) -> str:
    return ICONS.get(name, "").format(s=size)


# ========== 自定义 CSS ==========
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter",
                 "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    -webkit-font-smoothing: antialiased;
}
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { visibility: hidden; }

/* 保持侧边栏折叠/展开按钮可见 */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: block !important;
    opacity: 1 !important;
}

/* 侧边栏 */
[data-testid="stSidebar"] {
    background-color: #F7F8FA;
    border-right: 1px solid #E8E9EC;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    padding-top: 1.25rem;
    padding-left: 0.75rem;
    padding-right: 0.75rem;
}
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
    color: #4B5563 !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    padding: 0.5rem 0.65rem !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #EDEEF1 !important;
    color: #111827 !important;
}

/* Logo */
.brand {
    display: flex; align-items: center; gap: 10px;
    padding: 0.5rem 0.5rem 0.75rem;
}
.brand-logo {
    width: 32px; height: 32px;
    border-radius: 9px;
    background: #1E3A5F;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.brand-logo svg { stroke: #FFFFFF; }
.brand-name {
    font-size: 0.95rem; font-weight: 600;
    color: #111827; letter-spacing: -0.01em;
}

/* 侧栏分组标题 */
.sidebar-section {
    font-size: 0.72rem; font-weight: 500;
    color: #9CA3AF; text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.5rem 0 0.5rem;
    padding-left: 0.65rem;
}
.hist-item {
    display: flex; align-items: center; gap: 10px;
    padding: 0.45rem 0.65rem;
    border-radius: 8px;
    color: #4B5563; font-size: 0.86rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.hist-item svg { stroke: #9CA3AF; flex-shrink: 0; }

/* 主区 */
.main .block-container {
    max-width: 100%;
    padding: 1.5rem 2rem 6rem;
}
.content-narrow { max-width: 920px; margin: 0 auto; }

/* 欢迎页 */
.welcome-wrap {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding-top: 6vh;
}
.welcome-avatar {
    width: 56px; height: 56px;
    border-radius: 16px;
    background: #1E3A5F;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 1.5rem;
}
.welcome-avatar svg { stroke: #FFFFFF; }
.welcome-title {
    font-size: 1.75rem; font-weight: 600;
    color: #111827; letter-spacing: -0.02em;
    margin-bottom: 0.5rem; text-align: center;
}
.welcome-sub {
    font-size: 0.95rem; color: #6B7280;
    margin-bottom: 3rem; text-align: center;
}

/* 快捷卡片 */
.feature-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1rem; margin-bottom: 2.5rem;
}
.feature-card {
    background: #FFFFFF;
    border: 1px solid #E8E9EC;
    border-radius: 14px;
    padding: 1.5rem 1.25rem 1.35rem;
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
}
.feature-card:hover {
    transform: translateY(-2px);
    border-color: #D1D5DB;
    box-shadow: 0 8px 24px rgba(17,24,39,0.07);
}
.feature-icon {
    width: 38px; height: 38px;
    border-radius: 10px;
    background: #F3F4F6;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 1rem;
}
.feature-icon svg { stroke: #374151; }
.feature-title { font-size: 0.95rem; font-weight: 600; color: #111827; margin-bottom: 0.35rem; }
.feature-desc { font-size: 0.83rem; color: #6B7280; line-height: 1.5; }

/* 主区快捷按钮 */
.main .stButton > button {
    background: #FFFFFF !important;
    border: 1px solid #E8E9EC !important;
    border-radius: 10px !important;
    padding: 0.55rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #111827 !important;
    transition: all 0.16s ease !important;
    box-shadow: 0 1px 2px rgba(17,24,39,0.04) !important;
    width: 100% !important;
}
.main .stButton > button:hover {
    border-color: #D1D5DB !important;
    box-shadow: 0 6px 16px rgba(17,24,39,0.08) !important;
    transform: translateY(-1px) !important;
}

/* 输入框 */
.stTextInput input {
    background: #F3F4F6 !important;
    border: 1px solid #E8E9EC !important;
    border-radius: 12px !important;
    padding: 0.6rem 0.9rem !important;
    font-size: 0.93rem !important;
    color: #111827 !important;
    height: 42px !important;
}
.stTextInput input:focus {
    border-color: #111827 !important;
    box-shadow: 0 0 0 3px rgba(17,24,39,0.08) !important;
}

/* 发送按钮（form_submit_button） */
.main [data-testid="stFormSubmitButton"] > button {
    background: #1E3A5F !important;
    color: #FFFFFF !important;
    border: none !important;
    height: 44px !important;
    padding: 0 1.5rem !important;
    font-weight: 500 !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    white-space: nowrap !important;
    box-shadow: none !important;
    transition: background 0.15s ease !important;
}
.main [data-testid="stFormSubmitButton"] > button:hover {
    background: #16304F !important;
}

/* 聊天气泡 */
.chat-row { display: flex; gap: 10px; align-items: flex-start; width: 100%; margin-bottom: 1rem; }
.chat-row.user { flex-direction: row-reverse; }
.avatar {
    width: 30px; height: 30px;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 0.78rem; font-weight: 600;
}
.avatar.user { background: #1E3A5F; color: #FFFFFF; }
.avatar.assistant { background: #F3F4F6; }
.avatar.assistant svg { stroke: #374151; }

/* 报告区：更像正式文档 */
.bubble {
    max-width: 82%; padding: 1.25rem 1.5rem;
    border-radius: 14px; font-size: 0.92rem;
    line-height: 1.7; word-break: break-word;
}
.bubble.user { background: #1E3A5F; color: #FFFFFF; border-bottom-right-radius: 4px; }
.bubble.assistant {
    background: #F3F4F6; color: #111827;
    border: 1px solid #E8E9EC; border-bottom-left-radius: 4px;
}
.bubble h1 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
    margin: 0.2rem 0 0.9rem 0 !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid #E8E9EC !important;
}
.bubble h2 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #1E3A5F !important;
    margin: 1.2rem 0 0.5rem 0 !important;
}
.bubble h3 {
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
    margin: 1rem 0 0.4rem 0 !important;
}
.bubble p {
    font-size: 0.92rem !important;
    line-height: 1.75 !important;
    margin: 0.5rem 0 !important;
    color: #374151 !important;
}
.bubble ul, .bubble ol {
    font-size: 0.92rem !important;
    line-height: 1.75 !important;
    padding-left: 1.3rem !important;
    margin: 0.5rem 0 !important;
    color: #374151 !important;
}
.bubble li { margin: 0.3rem 0 !important; }
.bubble strong { font-weight: 600 !important; color: #111827 !important; }
.bubble blockquote {
    border-left: 3px solid #1E3A5F !important;
    padding-left: 0.8rem !important;
    color: #6B7280 !important;
    margin: 0.6rem 0 !important;
}
.bubble table {
    font-size: 0.85rem !important;
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 0.8rem 0 !important;
}
.bubble th, .bubble td {
    padding: 0.45rem 0.7rem !important;
    border: 1px solid #E8E9EC !important;
    text-align: left !important;
}
.bubble th {
    background: #F7F8FA !important;
    font-weight: 600 !important;
    color: #111827 !important;
}
.bubble code {
    background: #F3F4F6 !important;
    padding: 0.15rem 0.4rem !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    color: #1E3A5F !important;
}
.bubble hr {
    border: none !important;
    border-top: 1px solid #E8E9EC !important;
    margin: 1rem 0 !important;
}

/* 资源视图：expander 内的 Markdown 字号收敛 */
[data-testid="stExpander"] h1 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    margin: 0.6rem 0 0.4rem 0 !important;
    color: #111827 !important;
}
[data-testid="stExpander"] h2 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    margin: 0.6rem 0 0.35rem 0 !important;
    color: #1E3A5F !important;
}
[data-testid="stExpander"] h3 {
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    margin: 0.5rem 0 0.3rem 0 !important;
    color: #111827 !important;
}
[data-testid="stExpander"] p {
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
    margin: 0.35rem 0 !important;
    color: #374151 !important;
}
[data-testid="stExpander"] ul,
[data-testid="stExpander"] ol {
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
    padding-left: 1.3rem !important;
    margin: 0.35rem 0 !important;
    color: #374151 !important;
}
[data-testid="stExpander"] li {
    margin: 0.2rem 0 !important;
}
[data-testid="stExpander"] strong {
    font-weight: 600 !important;
    color: #111827 !important;
}

svg { vertical-align: middle; }
</style>
""", unsafe_allow_html=True)


# ========== Session State（多会话结构） ==========
def new_session():
    """创建一个新会话"""
    return {
        "id": str(uuid.uuid4()),
        "title": "新对话",
        "messages": [],
    }


def init_state():
    defaults = {
        "lang": "zh",
        "view": "chats",
        "sessions": [new_session()],       # 会话列表
        "current_session_id": None,        # 当前会话 id
        "pending_query": None,             # 待发送的问题
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # 初始化 current_session_id
    if st.session_state.current_session_id is None:
        st.session_state.current_session_id = st.session_state.sessions[0]["id"]

init_state()


def get_current_session():
    """获取当前会话对象"""
    for s in st.session_state.sessions:
        if s["id"] == st.session_state.current_session_id:
            return s
    return st.session_state.sessions[0]


lang = st.session_state.lang
view = st.session_state.view
current_session = get_current_session()


# ========== 侧边栏 ==========
with st.sidebar:
    # Logo
    st.markdown(
        f'<div class="brand">'
        f'<div class="brand-logo">{icon("trend", 18)}</div>'
        f'<span class="brand-name">{t("app_title", lang)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 新建对话
    if st.button(f"＋  {t('new_chat', lang)}", key="new_chat", use_container_width=True):
        new_s = new_session()
        st.session_state.sessions.append(new_s)
        st.session_state.current_session_id = new_s["id"]
        st.session_state.view = "chats"
        st.rerun()

    st.markdown(
        f'<div class="sidebar-section">{t("workspace", lang)}</div>',
        unsafe_allow_html=True,
    )

    nav_items = [
        ("chats", t("chats", lang)),
        ("exports", t("exports", lang)),
        ("sources", t("sources", lang)),
    ]
    for key, label in nav_items:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.view = key
            st.rerun()

    st.markdown(
        f'<div class="sidebar-section">{t("recent_chats", lang)}</div>',
        unsafe_allow_html=True,
    )

    # 最近会话列表（倒序显示，最多 8 个）
    sessions_sorted = list(reversed(st.session_state.sessions))
    if not sessions_sorted:
        st.markdown(
            f'<div class="hist-item" style="color:#9CA3AF;font-size:0.83rem">'
            f'{icon("clock", 14)}  {t("no_history", lang)}</div>',
            unsafe_allow_html=True,
        )
    else:
        for i, s in enumerate(sessions_sorted[:8]):
            is_active = (s["id"] == st.session_state.current_session_id)
            marker = "● " if is_active else ""
            # 用按钮切换会话
            btn_label = f'{marker}{s["title"][:20]}'
            if st.button(btn_label, key=f"sess_{s['id']}", use_container_width=True):
                st.session_state.current_session_id = s["id"]
                st.session_state.view = "chats"
                st.rerun()

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 底部设置 / 帮助
    col1, col2 = st.columns(2)
    with col1:
        if st.button(t("settings", lang), key="nav_settings", use_container_width=True):
            st.session_state.view = "settings"
            st.rerun()
    with col2:
        if st.button(t("help", lang), key="nav_help", use_container_width=True):
            st.session_state.view = "help"
            st.rerun()


# ========== 主区 ==========

if view == "chats":
    # 当前会话没有消息 → 显示欢迎页
    if not current_session["messages"]:
        st.markdown(
            '<div class="welcome-wrap">'
            '<div class="welcome-avatar">' + icon("trend", 26) + '</div>'
            f'<div class="welcome-title">{t("welcome", lang)}</div>'
            f'<div class="welcome-sub">{t("welcome_sub", lang)}</div>'
            '<div class="content-narrow" style="width:100%">'
            '<div class="feature-grid">'
            '<div class="feature-card">'
            '<div class="feature-icon">' + icon("bar-chart-3", 19) + '</div>'
            f'<div class="feature-title">{t("quick_query", lang)}</div>'
            f'<div class="feature-desc">{t("quick_query_desc", lang)}</div>'
            '</div>'
            '<div class="feature-card">'
            '<div class="feature-icon">' + icon("search", 19) + '</div>'
            f'<div class="feature-title">{t("quick_attribution", lang)}</div>'
            f'<div class="feature-desc">{t("quick_attribution_desc", lang)}</div>'
            '</div>'
            '<div class="feature-card">'
            '<div class="feature-icon">' + icon("file-text", 19) + '</div>'
            f'<div class="feature-title">{t("quick_policy", lang)}</div>'
            f'<div class="feature-desc">{t("quick_policy_desc", lang)}</div>'
            '</div>'
            '</div></div></div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(3)
        with cols[0]:
            if st.button(t("quick_query", lang), key="q1", use_container_width=True):
                st.session_state.pending_query = "华南区8月利润是多少？"
                st.rerun()
        with cols[1]:
            if st.button(t("quick_attribution", lang), key="q2", use_container_width=True):
                st.session_state.pending_query = "为什么上个月华南区利润下降了？"
                st.rerun()
        with cols[2]:
            if st.button(t("quick_policy", lang), key="q3", use_container_width=True):
                st.session_state.pending_query = "差旅住宿费报销的标准是什么？"
                st.rerun()

    # 显示当前会话的消息
    for msg in current_session["messages"]:
        role = msg["role"]

        # 根据角色生成头像
        if role == "user":
            av = '<div class="avatar user">U</div>'
        else:
            av = f'<div class="avatar assistant">{icon("sparkles", 15)}</div>'

        st.markdown(
            f'<div class="chat-row {role}">'
            f'{av}'
            f'<div class="bubble {role}">{msg["content"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 如果有图表，渲染出来
        if msg.get("chart_json"):
            import plotly.io as pio

            fig = pio.from_json(msg["chart_json"])
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"hist_chart_{id(msg)}",
            )

    # 输入框（form 方案：提交后自动清空）
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        input_cols = st.columns([5.5, 1])
        with input_cols[0]:
            draft = st.text_input(
                "draft_input",
                label_visibility="collapsed",
                placeholder=t("input_placeholder", lang),
            )
        with input_cols[1]:
            send_clicked = st.form_submit_button(
                t("send", lang),
                use_container_width=True,
            )

    if send_clicked and draft.strip():
        st.session_state.pending_query = draft.strip()
        st.rerun()

    if st.session_state.get("pending_query"):
        user_input = st.session_state.pending_query
        st.session_state.pending_query = None
    else:
        user_input = None

    # 调用真实后端
    if user_input:
        current_session["messages"].append({"role": "user", "content": user_input})
        if current_session["title"] == "新对话":
            current_session["title"] = user_input[:24]

        with st.chat_message("assistant"):
            with st.status("正在分析...", expanded=False) as status:
                graph = get_graph()
                result = graph.invoke({
                    "messages": [HumanMessage(content=user_input)],
                    "sql_retry_count": 0,
                })

                intent = result.get("intent")
                if intent:
                    st.write(f"意图识别：{intent}")
                if result.get("generated_sql"):
                    st.write("SQL 已生成并执行")
                if result.get("business_context"):
                    st.write("知识库检索完成")
                status.update(label="分析完成", state="complete")

            # ===== 文字回答 =====
            response = result.get("final_answer") or "（未生成回答）"
            st.markdown(response)

            # ===== 图表：紧跟文字之后 =====
            if result.get("chart_json"):
                import plotly.io as pio

                fig = pio.from_json(result["chart_json"])
                st.plotly_chart(fig, use_container_width=True)

        current_session["messages"].append({
            "role": "assistant",
            "content": response,
            "chart_json": result.get("chart_json", ""),
        })
        st.rerun()


elif view == "exports":
    st.markdown(f"## {t('exports', lang)}")
    st.info(f"*{t('no_exports', lang)}*")


elif view == "sources":
    st.markdown(f"## {t('sources', lang)}")
    tab1, tab2 = st.tabs([t("data_source", lang), t("knowledge_base", lang)])

    with tab1:
        # ===== 数据源：从 SQLite 动态读取 =====
        st.caption(t("read_only", lang))
        import sqlite3
        conn = sqlite3.connect("data/financial.db")
        c = conn.cursor()

        # 表列表
        c.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%'
        """)
        tables = [r[0] for r in c.fetchall()]

        for tbl in tables:
            c.execute(f"PRAGMA table_info({tbl})")
            cols = c.fetchall()
            c.execute(f"SELECT COUNT(*) FROM {tbl}")
            row_count = c.fetchone()[0]

            st.markdown(f"**表：`{tbl}`**")
            st.caption(f"字段数：{len(cols)}　行数：{row_count}")
            col_df = [{"字段": r[1], "类型": r[2]} for r in cols]
            st.dataframe(col_df, use_container_width=True, hide_index=True)

        conn.close()

    with tab2:
        st.caption(t("read_only", lang))
        import glob
        import os

        kb_root = "data/knowledge_base"
        categories = sorted([
            d for d in os.listdir(kb_root)
            if os.path.isdir(os.path.join(kb_root, d))
        ])

        if not categories:
            st.info("知识库为空")
        else:
            for cat in categories:
                files = sorted(glob.glob(os.path.join(kb_root, cat, "*.md")))
                st.markdown(
                    f"**{cat}**　<span style='color:#9CA3AF;font-size:0.82rem'>"
                    f"（{len(files)} 个文档）</span>",
                    unsafe_allow_html=True,
                )
                for f in files:
                    fname = os.path.basename(f)
                    with st.expander(f" {fname}"):
                        with open(f, "r", encoding="utf-8") as fp:
                            content = fp.read()
                        st.markdown(content)
                st.markdown("")


elif view == "settings":
    st.markdown(f"## {t('settings', lang)}")
    st.markdown(f"### {t('settings_language', lang)}")
    lang_choice = st.radio(
        "Language",
        options=["中文", "English"],
        index=0 if st.session_state.lang == "zh" else 1,
        horizontal=True,
    )
    if lang_choice == "中文" and st.session_state.lang != "zh":
        st.session_state.lang = "zh"
        st.rerun()
    elif lang_choice == "English" and st.session_state.lang != "en":
        st.session_state.lang = "en"
        st.rerun()
    st.markdown(t("settings_about_text", lang))


elif view == "help":
    st.markdown(f"## {t('help', lang)}")
    st.markdown(t("help_text", lang))