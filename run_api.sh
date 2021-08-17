#!/bin/bash
sudo docker build -t ml_api .
sudo docker run -p 8080:1313 ml_api:latest
