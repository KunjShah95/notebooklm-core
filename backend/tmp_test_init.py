import sys
sys.path.append(r"c:\notebooklm core\backend")
from ui.app import initialize_components
print('Calling initialize_components...')
try:
    rag = initialize_components(provider='groq', model_name='llama3-8b-8192')
    print('initialize_components returned:', type(rag))
except Exception as e:
    print('Exception during initialization:', e)
