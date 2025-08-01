# YOLO Labelling Pipeline

## Abstract
Gemini 2.0 and Grounding Dino both show outstanding performance in object detection. In addition, they respond very well to user prompts, making them extremely flexible and user friendly when deployed in related projects. This repo  uses streamlit to implement a user accesible framework to connect to their features.

## Requirements
* Docker
* Docker Compose
* GroundingDino ([*groundingdino_swint_ogc.pth*](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth))
* LangChain
* minIO

## Set-up
1. Load pretrain weights and api-keys to the repository.
```Shell
#Recommanded directory structure
├── docker-compose.yml
├── gemini2
│   ├── Dockerfile
│   ├── gemini_pipeline_api.py
│   ├── image_utils.py
│   ├── langchain_client_backend.py
│   └── requirements.txt
├── grounding_dino
│   ├── Dockerfile
│   ├── environment.yaml
│   ├── groudingdino_pipeline_api.py
│   ├── groundingdino
│   ├── LICENSE
│   ├── README.md
│   ├── requirements.txt
│   └── weights
│       └── groundingdino_swint_ogc.pth
├── README.md
└── st_app
    ├── Dockerfile
    ├── labellingpipeline_st.py
    ├── requirements.txt
    └── storage.py
````
2. Have docker compose up and running and access the streamlit application directly.
```Shell
> docker compose up -D
> direct to http://localhost:8501
```
3. After uploading the base image and selecting a desired model,  enter the object to be detected in the YOLO requests text box. Hit generate to obtain a labelled image.