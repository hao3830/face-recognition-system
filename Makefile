start: 
	python3 -m uvicorn main:app --port 8000 --host 0.0.0.0

build:
	docker-compose up --build -d

.PHONY: start build