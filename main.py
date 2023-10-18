from faster_whisper import WhisperModel
import streamlit as st
import datetime
from io import BytesIO

st.set_page_config(
    page_title="Word Wizard App",
    page_icon="✏️",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.markdown(
    '''
    whisperを使ってwebで文字起こしができるアプリになります。
    ''')

uploaded_file = st.file_uploader("音声ファイル", type=["mp4", "mp3", "wav"])
on = st.toggle("動画または音声の再生")

file_flag = False
download_flag = False
texts = []

if uploaded_file is not None:
    # st.json({'filename': uploaded_file.name, 'filetype': uploaded_file.type, 'filesize': uploaded_file.size})
    upload_bytes = BytesIO(uploaded_file.getvalue())
    if on:
        if "mp4" in uploaded_file.name:
            st.video(upload_bytes)
        else:
            st.audio(upload_bytes)
    file_flag = True
        
col1, col2 = st.columns(2)
    
with col1:
    start = st.button("文字起こしスタート")
    if start and file_flag:
        # placeholder_result = st.empty()
        container = st.container()
        with st.spinner('モデルの読込...'):
            # model_size = "large-v2"
            model_size = "medium"
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            # placeholder_result.success('モデルの読込完了')
            container.success('モデルの読込完了')
        
        with st.spinner('文字起こし中...'):

            segments, info = model.transcribe(upload_bytes, beam_size=5, vad_filter=True)

            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            
            placeholder = st.empty()
            with placeholder.container():
                for segment in segments:
                    out = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
                    st.write(out)
                    texts.append(out)
                else:
                    # placeholder_result.success('文字起こし完了')
                    container.success('文字起こし完了')
                    download_flag = True
    else:
        pass
                    
with col2:
    if download_flag:
        now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        title = st.text_input("保存ファイル名", now)
        st.write("保存ファイル名が無ければ、日時名になります")
        
        text = "\n".join(texts)
        download_two = st.download_button("ダウンロード", text, f"{title}.txt")
    else:
        st.info("文字起こしが完了するとダウンロードボタンが出現します")
        
st.divider()
st.markdown(
    '''
    [X](https://twitter.com/siro020)　[GitHub](https://github.com/massao000)
    ''')