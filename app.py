import streamlit as st
import pandas as pd
from transformers import pipeline
import altair as alt # Grafik renklendirme için gerekli

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="Müşteri İçgörü Paneli", page_icon="🛍️", layout="wide")

# --- 2. MODELİ YÜKLEME ---
@st.cache_resource
def model_yukle():
    try:
        return pipeline("sentiment-analysis", model="my_sentiment_model", tokenizer="my_sentiment_model")
    except Exception as e:
        return None

classifier = model_yukle()

# --- 3. YAN MENÜ ---
st.sidebar.title("🛍️ E-Ticaret Analizi")
if classifier:
    st.sidebar.success("Model Yüklendi ✅")
else:
    st.sidebar.error("Model Bulunamadı ❌")
    
st.sidebar.info("Model: BERTurk (Fine-Tuned)")
st.sidebar.write("Bu panel, müşteri yorumlarını yapay zeka ile analiz eder.")

# --- 4. ANA BAŞLIK ---
st.title("📊 İnteraktif Müşteri İçgörü Paneli")
st.markdown("Türkçe e-ticaret yorumları için duygu analizi ve istatistiksel gösterge paneli.")

# --- 5. SEKME TANIMLARI ---
tab1, tab2 = st.tabs(["🔍 Canlı Analiz (Demo)", "📈 Genel İçgörüler (Rapor)"])

# ---------------- SEKME 1: CANLI TEST ----------------
with tab1:
    st.subheader("Tekil Yorum Analizi")
    
    # Hafıza (Session State)
    if 'text_input' not in st.session_state:
        st.session_state['text_input'] = ""

    def yazi_guncelle(yeni_yazi):
        st.session_state['text_input'] = yeni_yazi

    # Hazır Örnek Butonları
    st.markdown("##### Hızlı deneme yapmak için bir örneğe tıklayın:")
    col_b1, col_b2, col_b3 = st.columns(3)
    
    col_b1.button("📝 Örnek 1 (Pozitif)", on_click=yazi_guncelle, args=["Ürün harika, paketleme çok özenliydi. Teşekkürler!"])
    col_b2.button("📝 Örnek 2 (Nötr)", on_click=yazi_guncelle, args=["Fena değil, fiyatına göre idare eder ama kargo gecikti."])
    col_b3.button("📝 Örnek 3 (Negatif)", on_click=yazi_guncelle, args=["Berbat bir ürün, sakın almayın paranıza yazık."])

    # Kullanıcı Giriş Alanı
    yorum_metni = st.text_area("Analiz edilecek yorumu giriniz:", key="text_input", height=100)
    
    if st.button("ANALİZ ET", type="primary"):
        if classifier and yorum_metni:
            sonuc = classifier(yorum_metni[:512])[0]
            label = sonuc['label']
            score = sonuc['score']
            
            col1, col2 = st.columns(2)
            with col1:
                if label == "LABEL_2":
                    st.success("Sonuç: **POZİTİF (Mutlu Müşteri)** 😊")
                elif label == "LABEL_1":
                    st.warning("Sonuç: **NÖTR (Kararsız)** 😐")
                else:
                    st.error("Sonuç: **NEGATİF (Mutsuz Müşteri)** 😡")
            with col2:
                st.metric("Model Güven Skoru", f"%{score*100:.2f}")
        elif not classifier:
            st.error("Model yüklenemedi!")
        else:
            st.warning("Lütfen bir metin girin.")

# ---------------- SEKME 2: İÇGÖRÜLER (Renkli Versiyon) ----------------
with tab2:
    st.subheader("📊 Geçmiş Veri Analizi ve Genel İstatistikler")
    
    try:
        df = pd.read_csv("test.csv")
        df['Duygu'] = df['label'].map({0: 'Negatif', 1: 'Nötr', 2: 'Pozitif'})
        
        # --- METRİKLER (KPIs) ---
        total = len(df)
        pos = len(df[df['Duygu']=='Pozitif'])
        neu = len(df[df['Duygu']=='Nötr'])
        neg = len(df[df['Duygu']=='Negatif'])
        
        pos_oran = (pos / total) * 100
        neu_oran = (neu / total) * 100
        neg_oran = (neg / total) * 100
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Toplam Yorum", f"{total}")
        k2.metric("Memnuniyet", f"{pos}", f"%{pos_oran:.1f} Oran", delta_color="normal")
        # Nötr gri kalsın (off)
        k3.metric("Kararsız", f"{neu}", f"%{neu_oran:.1f} Oran", delta_color="off")
        # Negatif artık KIRMIZI olacak (inverse)
        k4.metric("Şikayet", f"{neg}", f"%{neg_oran:.1f} Oran", delta_color="inverse")
        
        st.divider()
        
        # --- GRAFİKLER VE DETAYLAR ---
        col_g1, col_g2 = st.columns([1, 2]) 
        
        with col_g1:
            st.markdown("##### 📉 Duygu Dağılımı")
            
            # --- ÖZEL RENKLİ GRAFİK ---
            # Veriyi hazırla
            chart_data = df['Duygu'].value_counts().reset_index()
            chart_data.columns = ['Duygu', 'Adet']
            
            # Renkleri Belirle (Yeşil, Sarı, Kırmızı)
            renkler = alt.Scale(domain=['Pozitif', 'Nötr', 'Negatif'],
                                range=['#28a745', '#ffc107', '#dc3545']) # Yeşil, Sarı, Kırmızı
            
            # Grafiği Çiz
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Duygu', sort=['Pozitif', 'Nötr', 'Negatif']),
                y='Adet',
                color=alt.Color('Duygu', scale=renkler, legend=None),
                tooltip=['Duygu', 'Adet']
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
            
            st.markdown("##### 🏆 Popüler Kelimeler")
            st.info("Kargo, Paketleme, Kalite, Hızlı, Teşekkürler, Tavsiye") 
            
        with col_g2:
            st.markdown("##### 📝 Tüm Müşteri Yorumları")
            
            filtre = st.selectbox("Görüntülenecek Yorum Tipi:", ["Tümü", "Pozitif", "Negatif", "Nötr"])
            
            if filtre == "Tümü":
                gosterilecek_df = df
            else:
                gosterilecek_df = df[df['Duygu'] == filtre]
            
            st.dataframe(gosterilecek_df[['text', 'Duygu']].head(50), hide_index=True, use_container_width=True)
            
    except FileNotFoundError:
        st.error("Veri dosyası (test.csv) bulunamadı!")