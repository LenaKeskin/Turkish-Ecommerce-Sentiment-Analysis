#  Türkçe E-Ticaret Yorumlarında BERT Tabanlı Duygu Analizi: Müşteri İçgörü Paneli

##  Proje Özeti
Bu proje, Türkiye'nin önde gelen e-ticaret platformlarından (Hepsiburada) elde edilen Türkçe kullanıcı yorumlarını **Doğal Dil İşleme (NLP)** ve **Derin Öğrenme** teknikleriyle analiz ederek, 
işletmelere interaktif ve aksiyon alınabilir **müşteri içgörüleri** sunan uçtan uca bir yapay zeka çözümüdür. `BERTurk` modeli sayesinde yorumların duygu durumunu yüksek doğrulukla tespit etmekle kalmayıp, 
`Streamlit` tabanlı interaktif bir Dashboard ile bu analizleri görsel ve anlaşılır bir formata dönüştürmektedir.

**Problem Çözücü Yaklaşımım:**
Binlerce yorumu manuel okuma zorluğunu ortadan kaldırarak, markaların müşteri memnuniyetini anlık izlemesini ve ürün/hizmet iyileştirmelerini hızlıca yapmasını sağlamak.

## 🎥 Proje Demosu

Projenin nasıl çalıştığını, canlı analiz ekranını ve genel içgörüler panelini aşağıdaki videodan izleyebilirsiniz.

https://github.com/user-attachments/assets/demo.mp4
*(Not: Eğer video yukarıda otomatik oynamazsa, dosyalar arasındaki 'demo.mp4' dosyasına tıklayarak izleyebilirsiniz.)*

> **Alternatif İzleme:** [🎥 Tanıtım Videosunu İndir/İzle](demo.mp4)

##  Projenin Temel Amaçları ve Başarıları

*   **Veri Mühendisliği:** Ham ve gürültülü (etiket hataları içeren) e-ticaret verisini temizleyip, 15.000 satırlık dengeli ve yüksek kaliteli bir veri setine dönüştürmek.
*   **Modern NLP Model Geliştirme:** Türkçe'nin yapısal karmaşıklığına uygun, Transformer mimarisine sahip **BERTurk** modeliyle %97.90 gibi yüksek bir doğrulukla duygu analizi yapmak.
*   **Karşılaştırmalı Analiz:** Geleneksel yöntemlere (TF-IDF + Lojistik Regresyon) kıyasla BERTurk'ün başarıdaki **5 katlık** hata azalmasını bilimsel metriklerle ispatlamak.
*   **Ürünleştirme (Deployment):** Geliştirilen yapay zeka modelini, son kullanıcının kolayca etkileşime geçebileceği interaktif bir **Streamlit Dashboard**'a dönüştürmek.

*   ##  Kullanılan Teknolojiler

*   **Python:** Projenin ana geliştirme dili.
*   **HuggingFace Transformers:** BERTurk modelinin indirilmesi, ince ayarı (Fine-Tuning) ve yönetimi.
*   **Streamlit:** Veri analizlerini görselleştiren ve modelle etkileşimi sağlayan interaktif web arayüzü (Dashboard) geliştirme.
*   **Pandas:** Veri manipülasyonu ve ön işleme.
*   **Scikit-learn:** Baseline model (TF-IDF + Lojistik Regresyon) oluşturma ve metrik hesaplama.
*   **Altair & Matplotlib/Seaborn:** Veri görselleştirme ve grafik oluşturma.
*   **Google Colab:** GPU destekli model eğitimi için kullanılmıştır.

*   ##  Proje Adımları ve Elde Edilen Bulgular

1.  **Veri Temizliği ve Hazırlığı:**
    *   300.000+ satırlık ham Hepsiburada verisi temizlendi.
    *   Eksik/tekrarlayan veriler ve etiket hataları giderildi.
    *   1'den 5'e kadar puan dağılımını dengelemek için **Oversampling** tekniği kullanılarak 15.000 satırlık nihai veri seti oluşturuldu.
    *   Metinler küçük harfe çevrildi, noktalama ve sayılar temizlendi (BERTurk'e uygun).

2.  **Modelleme ve Eğitim:**
    *   **Baseline Model:** TF-IDF ve Lojistik Regresyon ile %92.49 doğruluk elde edildi.
    *   **Ana Model:** BERTurk modeli 3 epoch boyunca eğitilerek **%97.90** doğruluk oranına ulaştı.
    *   **Sonuç:** BERTurk, geleneksel modele göre hata oranını **5 kattan fazla** azaltarak üstünlüğünü kanıtladı. Özellikle "Ürün güzel ama kargo kötü" gibi bağlamsal ifadeleri başarıyla yorumladı.

3.  **İnteraktif Dashboard Geliştirme:**
    *   Modelin anlık tahmin yapabildiği "Canlı Analiz" sekmesi.
    *   Geçmiş veri istatistiklerini (Memnuniyet, Şikayet oranları, Duygu Dağılımı) görselleştiren "Genel İçgörüler" sekmesi.
    *   Nötr ve Negatif yorumları filtreleyerek gelişim alanlarını gösteren tablo.

## 🚀 Kurulum ve Çalıştırma Rehberi

Bu proje dosyaları, çalışmak için gerekli olan temel kodları ve veri setini içerir.

### ⚠️ Önemli Bilgilendirme (Model Dosyası Hakkında)
Eğitilen **BERTurk model dosyaları (~450 MB)** GitHub dosya boyutu sınırını aştığı için bu depoya (repository) doğrudan eklenememiştir.
*   Projenin çalışma mantığını ve çıktılarını yukarıdaki **Demo Videosu** üzerinden inceleyebilirsiniz.
*   Projeyi yerel bilgisayarınızda çalıştırmak isterseniz, `my_sentiment_model` klasörünü ayrıca temin etmeniz veya eğitmeniz gerekmektedir.

### Yerel Kurulum Adımları (Standart Prosedür)

**1. Repoyu Klonlayın:**
```bash
git clone https://github.com/KULLANICI_ADINIZ/Turkish-Ecommerce-Sentiment-Analysis.git
cd Turkish-Ecommerce-Sentiment-Analysis
2. Gerekli Kütüphaneleri Kurun:
code
Bash
pip install -r requirements.txt
3. Uygulamayı Başlatın:
code
Bash
streamlit run app.py
