from faster_whisper import WhisperModel
import streamlit as st
import datetime
from io import BytesIO

st.markdown(
    '''
    test
    ''')

uploaded_file = st.file_uploader("音声ファイル", type=["mp4", "mp3", "wav"])

file_flag = False
download_flag = False
texts = []

if uploaded_file is not None:
    # st.json({'FileName': uploaded_file.name, 'FileType': uploaded_file.type, 'FileSize': uploaded_file.size})
    upload_binary = BytesIO(uploaded_file.getvalue())
    st.write(upload_binary)
    file_flag = True
        
col1, col2 = st.columns(2)
    
with col1:
    start = st.button("文字起こしスタート")
    if start and file_flag:
        with st.spinner('モデルの読込...'):
            # model_size = "large-v2"
            model_size = "medium"
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            st.success('モデルの読込完了')
            st.toast('モデルの読込完了')
        
        with st.spinner('文字起こし中...'):

            segments, info = model.transcribe(upload_binary, beam_size=5, vad_filter=True)

            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            
            placeholder = st.empty()
            with placeholder.container():
                for segment in segments:
                    out = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
                    st.write(out)
                    texts.append(out)
                else:
                    st.success('Done!')
                    download_flag = True
    else:
        pass
                    
with col2:
    if download_flag:
        t = "\n".join(texts)
        now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        download_two = st.download_button("ダウンロード", t, f"file_{now}.txt")
    else:
        st.info("文字起こしを開始するとダウンロードボタンが出現します")