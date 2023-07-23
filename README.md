e2eqa
==============================

This is a final project of statistical learning class

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- File README
    ├── data
    │   ├── processed      <- Dữ liệu đã qua bước tiền xử lý
    │   └── raw            <- Dữ liệu thô.
    │
    ├── models             <- Các checkpoint, models đã train.
    │
    ├── notebooks          <- Các file notebooks tiền xử lý dữ liệu, chạy lần lượt theo thứ tự từ trên xuống.
    │                         
    │                         
    │
    │
    ├── reports            <- Báo cáo.
    │   └── report.pdf     
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code để thực hiện inference
    │   │
    │   ├── features       <- Scripts để processing dữ liệu
    │   │   |── text_utils.py       <- Các hàm để tiền xử lý dữ liệu
    │   |   └── graph_utils.py      <- Tiền xử lý dữ liệu dạng graph cho qa model
    │   │
    │   └── models         <- Scripts để sử dụng các models đã train để predict
    │   │   │             
    │   │   ├── predict_model.py     <- Inference kết quả
    │   │   |── bm25_utils.py        <- Class BM25    
    │   │   |── pairwise_model.py    <- Class Pairwise model
    │   │   └── qa_model.py          <- Class QA model
    │   │
    │   └── app.py         <- Streamlit app để chạy inference
    │
    └── submission
        └── submission.csv <- File submission

--------
demo: 

<iframe width="560" height="315" src="https://www.youtube.com/embed/dDLh55HYrfg" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> 
