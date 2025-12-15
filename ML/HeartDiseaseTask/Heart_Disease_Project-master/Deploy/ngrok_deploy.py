from pyngrok import ngrok

# Open a tunnel to the Streamlit app port (default 8501)
public_url = ngrok.connect(8501)
print("Ngrok URL:", public_url)
