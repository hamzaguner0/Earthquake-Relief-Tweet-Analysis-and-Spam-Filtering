from pathlib import Path
import os
import streamlit as st
from deprem_nlp.model import load_model, score

st.set_page_config(page_title="Deprem NLP · Araştırma demosu", page_icon="🔎")
st.title("Deprem yardım çağrısı sınıflandırma")
st.warning("Araştırma demosu. Acil yardım yönlendirme aracı değildir; tahminler insan kontrolü gerektirir. Gerçek kişilerin özel bilgilerini girmeyin.")
root = Path(__file__).resolve().parent
backend = st.selectbox("Model", ["Lojistik regresyon", "Yerel DistilBERTurk"])
threshold = st.slider("Yardım çağrısı karar eşiği", 0.1, 0.9, 0.5, 0.05)
text = st.text_area("Örnek metin", placeholder="Sentetik bir yardım çağrısı yazın.", max_chars=5000)

@st.cache_resource
def local_baseline(path):
    return load_model(path)

@st.cache_resource
def local_transformer(path):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_safetensors=True)
    model.eval()
    return tokenizer, model

if st.button("Sınıflandır"):
    if not text.strip():
        st.info("Bir örnek metin girin.")
    else:
        try:
            if backend == "Lojistik regresyon":
                path = Path(os.environ.get("DEPREM_BASELINE_PATH", root / "artifacts/baseline.joblib"))
                bundle = local_baseline(str(path))
                probability = float(score(bundle, [text])[0])
                if bundle["report"]["evaluation_source"] == "synthetic_smoke_test":
                    st.caption("Bu model yalnızca sentetik kurulum verisiyle eğitilmiştir.")
            else:
                import torch
                path = Path(os.environ.get("DEPREM_MODEL_DIR", root / "fine_tuned_distilberturk"))
                tokenizer, model = local_transformer(str(path))
                encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    probability = float(torch.softmax(model(**encoded).logits, dim=-1)[0, 1])
                st.caption("Yerel tarihsel model: karma etiketli değerlendirme bağımsız başarı kanıtı sayılmaz. En fazla 128 token işlenir.")
            st.subheader("Yardım çağrısı adayı" if probability >= threshold else "Diğer içerik adayı")
            st.metric("Model skoru · sınıf 1", f"{probability:.3f}")
            st.caption("Bu skor doğrulanmış doğruluk veya kalibre edilmiş kesinlik değildir. Metin kaydedilmez; bu arayüz dış API çağrısı yapmaz.")
        except FileNotFoundError:
            st.error("Model bulunamadı. README içindeki sentetik eğitim adımıyla başlayın.")
        except ImportError:
            st.error("Yerel transformer için isteğe bağlı transformer bağımlılıklarını kurun.")
        except (OSError, ValueError):
            st.error("Yerel model yüklenemedi. Güvenilir model klasörünü ve etiket eşlemesini kontrol edin.")
