start: 
	python -m uvicorn main:app --port 8000 --host 0.0.0.0

.PHONY: start