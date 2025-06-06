#!/bin/bash

docker tag fraude-detection-train-service ilyesnajjari/train-service:latest
docker push ilyesnajjari/train-service:latest

docker tag fraude-detection-predict-service ilyesnajjari/predict-service:latest
docker push ilyesnajjari/predict-service:latest

docker tag fraude-detection-ingestion-service ilyesnajjari/ingestion-service:latest
docker push ilyesnajjari/ingestion-service:latest

docker tag fraude-detection-compare-service ilyesnajjari/compare-service:latest
docker push ilyesnajjari/compare-service:latest

docker tag fraude-detection-frontend ilyesnajjari/frontend:latest
docker push ilyesnajjari/frontend:latest

#chmod +x push_all.sh
#./push_all.sh