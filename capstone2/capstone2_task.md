# Capstone 2 Task

## Projects

The idea is that you now apply everything we learned so far yourself.

* Capstone 2

This is what you need to do for each project

* Think of a problem that's interesting for you and find a dataset for that
* Describe this problem and explain how a model could be used
* Prepare the data and doing EDA, analyze important features
* Train multiple models, tune their performance and select the best model
* Export the notebook into a script
* Put your model into a web service and deploy it locally with Docker
* Bonus points for deploying the service to the cloud

## Dataset

Student Stress Monitoring Datasets
Combination of Stress, Well-being Factors, Underlying Causes and Their Impacts

About Dataset
Understanding the Underlying Causes and Their Impact on Today's Students
About the Dataset:
This dataset investigates the root causes of stress among students, derived from a nationwide survey. It includes around 20 key features grouped under five scientifically identified categories:

🧠 Psychological Factors
anxiety_level
self_esteem
mental_health_history
depression
🏥 Physiological Factors
headache
blood_pressure
sleep_quality
breathing_problem
🌆 Environmental Factors
noise_level
living_conditions
safety
basic_needs
🎓 Academic Factors
academic_performance
study_load
teacher_student_relationship
future_career_concerns
🤝 Social Factors
social_support
peer_pressure
extracurricular_activities
bullying
Survey on Stress and Well-being Factors Among College Students (Ages 18–21)
📘 About the Dataset:
This dataset captures survey responses from 843 college students aged 18–21 regarding their experiences with stress, health, relationships, academics, and emotional well-being. The responses were collected via Google Forms using a five-point Likert scale ("Not at all" to "Extremely") and anonymized to protect privacy.

It enables nuanced analysis of emotional and physical stress indicators and their correlations with academic performance and lifestyle factors.

🔑 Key Features (Selected):
👤 Demographic
Gender: Coded as 0 (Male), 1 (Female)
Age: Numeric age (18 to 21)
🧠 Emotional and Stress Indicators
Have you recently experienced stress in your life?
Have you noticed a rapid heartbeat or palpitations?
Have you been dealing with anxiety or tension recently?
Do you face any sleep problems or difficulties falling asleep?
Do you have trouble concentrating on your academic tasks?
Have you been feeling sadness or low mood?
Do you get irritated easily?
Do you often feel lonely or isolated?
🩺 Physical and Health Indicators
Have you been getting headaches more often than usual?
Have you been experiencing any illness or health issues?
Have you gained/lost weight?
📚 Academic & Environment Stressors
Do you feel overwhelmed with your academic workload?
Are you in competition with your peers, and does it affect you?
Do you lack confidence in your academic performance?
Do you lack confidence in your choice of academic subjects?
Academic and extracurricular activities conflicting for you?
Do you attend classes regularly?
Are you facing any difficulties with your professors or instructors?
Is your working environment unpleasant or stressful?
Is your hostel or home environment causing you difficulties?
💬 Social & Relationship Factors
Do you find that your relationship often causes you stress?
Do you struggle to find time for relaxation and leisure activities?
📌 Target Variable
Which type of stress do you primarily experience?: Eustress, Distress, No Stress
📝 Citation
@article{ovi2025protecting,
  title={Protecting Student Mental Health with a Context-Aware Machine Learning Framework for Stress Monitoring},
  author={Ovi, Md Sultanul Islam and Hossain, Jamal and Rahi, Md Raihan Alam and Akter, Fatema},
  journal={arXiv preprint arXiv:2508.01105},
  year={2025}
}

File1 [Stress_Dataset.csv](./capstone2/Stress_Dataset.csv)
File2 [StressLevelDataset.csv](./capstone2/StressLevelDataset.csv)

## Deliverables

For a project, you repository/folder should contain the following:

* `README.md` with
  * Description of the problem
  * Instructions on how to run the project
* Data
  * You should either commit the dataset you used or have clear instructions how to download the dataset
* Notebook (suggested name - `notebook.ipynb`) with
  * Data preparation and data cleaning
  * EDA, feature importance analysis
  * Model selection process and parameter tuning
* Script `train.py` (suggested name)
  * Training the final model
  * Saving it to a file (e.g. pickle) or saving it with specialized software (BentoML)
* Script `predict.py` (suggested name)
  * Loading the model
  * Serving it via a web service (with Flask or specialized software - BentoML, KServe, etc)
* Files with dependencies
  * `Pipenv` and `Pipenv.lock` if you use Pipenv
  * or equivalents: conda environment file, requirements.txt or pyproject.toml
* `Dockerfile` for running the service
* Deployment
  * URL to the service you deployed or
  * Video or image of how you interact with the deployed service

## Evaluation Criteria

The project will be evaluated using these criteria:

* Problem description
  * 0 points: Problem is not described
  * 1 point: Problem is described in README birefly without much details
  * 2 points: Problem is described in README with enough context, so it's clear what the problem is and how the solution
will be used
* EDA
  * 0 points: No EDA
  * 1 point: Basic EDA (looking at min-max values, checking for missing values)
  * 2 points: Extensive EDA (ranges of values, missing values, analysis of target variable, feature importance analysis)
      For images: analyzing the content of the images.
      For texts: frequent words, word clouds, etc
* Model training
  * 0 points: No model training
  * 1 point: Trained only one model, no parameter tuning
  * 2 points: Trained multiple models (linear and tree-based).
      For neural networks: tried multiple variations - with dropout or without, with extra inner layers or without
  * 3 points: Trained multiple models and tuned their parameters.
      For neural networks: same as previous, but also with tuning: adjusting learning rate, dropout rate, size of the inner layer, etc.
* Exporting notebook to script
  * 0 points: No script for training a model
  * 1 point: The logic for training the model is exported to a separate script
* Reproducibility
  * 0 points: Not possitble to execute the notebook and the training script. Data is missing or it's not easiliy accessible
  * 1 point: It's possible to re-execute the notebook and the training script without errors. The dataset is committed in the project repository or there are clear instructions on how to download the data
* Model deployment
  * 0 points: Model is not deployed
  * 1 point: Model is deployed (with Flask, BentoML or a similar framework)
* Dependency and enviroment management
  * 0 points: No dependency management
  * 1 point: Provided a file with dependencies (requirements.txt, pipfile, bentofile.yaml with dependencies, etc)
  * 2 points: Provided a file with dependencies and used virtual environment. README says how to install the dependencies and how to
activate the env
* Containerization
  * 0 points: No containerization
  * 1 point: Dockerfile is provided or a tool that creates a docker image is used (e.g. BentoML)
  * 2 points: The application is containerized and the README describes how to build a container and how to run it
* Cloud deployment
  * 0 points: No deployment to the cloud
  * 1 point: Docs describe clearly (with code) how to deploy the service to cloud or kubernetes cluster (local or remote)
  * 2 points: There's code for deployment to cloud or kubernetes cluster (local or remote). There's a URL for testing - or video/screenshot of testing it

Total max 16 points

## FAQ

**Q**: Can I use poetry / virtual env for managing dependencies; catboost for boosting and FastAPI for creating a web service?

> Yes, you can use any library you want. But please make sure to document everything and clearly explain what you use.
> Think of your peers who will review it - they don't necessarily know what these libraries are.
> Please give them enough context to understand your project.
