# 🍺 Brew-AI: Fermentation Prediction & Comparison Engine

Brew-AI is a demo standalone desktop application designed for brewers, lab technicians, and fermentation
scientists. It leverages machine learning models to predict key outcomes of the beer fermentation process based on a
comprehensive set of 23 input variables.

This tool moves beyond simple prediction, allowing you to manage a library of models for different targets, compare
their performance side-by-side, and analyze your data, all within a polished and intuitive graphical interface that runs
locally in your web browser.

Available machine learning models are based on the paper ["Leveraging Periodicity for Tabular Deep Learning"](https://www.mdpi.com/2079-9292/14/6/1165), whose code can be found [here](https://github.com/matteo-rizzo/periodic-tabular-dl).

## ✨ Key Features

* **Multi-Target Prediction:** Predict different variables of the fermentation process (e.g., "Diacetyl Rest," "Final
  Gravity") by simply selecting the desired outcome from a dropdown.
* **Multi-Model Comparison:** For any single target, run and compare multiple trained models at once to evaluate their
  performance and prediction variance.
* **Hierarchical Model Management:** Simply drop your trained model files (`.pth`, `.pt`) into organized
  subdirectories (e.g., `models/Diacetyl Rest/`) and the app will automatically detect and load them.
* **Dual Input Methods:**
    * **Manual Form Entry:** A clean, organized form with collapsible sections allows for the quick prediction of a
      single batch using 23 specific process variables.
    * **Batch Prediction from CSV:** Upload a CSV file containing multiple batches to get predictions for all of them at
      once.
* **Standalone & Cross-Platform:** Packaged into a single executable folder, allowing anyone on macOS or Windows to run
  the application without needing to install Python or any libraries.

## 🚀 Getting Started (For End-Users)

If you have downloaded a packaged version of the application (e.g., `BrewAI_macOS.zip`), follow these steps to run it.

1. **Download:** Get the latest release ZIP file for your operating system.
2. **Unzip:** Extract the contents of the ZIP file to a convenient location on your computer (like your Desktop or
   Documents folder). This will create a folder named `BrewAI`.
3. **Launch the Server:** Open the `BrewAI` folder and double-click the executable file (`BrewAI.exe` on Windows,
   `BrewAI.app` on macOS).
    * The application will start a server in the background. **No window will appear immediately.** This is normal.
      Please allow 10-20 seconds for it to initialize.
4. **Open the Interface:** Manually open your web browser (Chrome, Safari, Firefox, etc.) and navigate to the following
   address:
   > **[http://127.0.0.1:7860](http://127.0.0.1:7860)**
5. The application interface will load in your browser. You can now begin your analysis\!

## 📋 How to Use the Application

The interface is designed to be a straightforward, step-by-step process controlled from the left sidebar.

1. **Select Target & Model(s):**

    * In the **Step 1** panel, use the first dropdown to select the variable you want to predict (e.g., "Diacetyl
      Rest").
    * The checkbox group below it will automatically populate with all the models available for that target. Select one
      or more models you wish to run.

2. **Provide Input Data:**

    * In the **Step 2** panel, choose your input method:
        * **Manual Entry:** Fill out the form with the 23 process variables for a single batch. The form is organized
          into collapsible sections for clarity.
        * **Batch from CSV:** Select this option and upload a CSV file. **Important:** The CSV file must contain columns
          whose names exactly match the required input variables.

3. **Run Prediction:**

    * Click the **"🚀 Run Prediction / Comparison"** button. A progress bar will show the status.

4. **Analyze Results:**

    * The results will appear in the **"📈 Results"** tab on the right.
    * **Single Model Run:** You will see a direct prediction value (for manual entry) or a new prediction column in your
      data (for batch CSV).
    * **Multi-Model Comparison:** The results will be shown in sub-tabs:
        * **Side-by-Side Data:** A table with a separate prediction column for each model.
        * **Statistics:** A summary table with descriptive statistics (mean, std, etc.) for each model's predictions.
        * **Distributions Plot:** A visual plot showing the density of predictions from all models, overlaid for easy
          comparison.

## 🛠️ For Developers & Contributors

This section provides instructions for setting up the project for development, adding new models, and building the
standalone installer.

### Setup & Installation

1. **Clone the Repository:**
   `git clone https://github.com/your-username/brewery-ml.git`
   `cd brewery-ml`

2. **Create and Activate a Virtual Environment:**
   `python3 -m venv .venv`
   `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)

3. **Install Dependencies:**
   `pip install -r requirements.txt`

### Running the App Locally

To run the application in development mode with live reloading:

`python3 app.py`

### Adding New Models

The application will automatically detect new models if you follow this structure:

1. **Identify the Target:** Determine the exact name of the variable your model predicts (e.g., "Diacetyl Rest").
2. **Create a Subdirectory:** Inside the `models/` directory, create a new folder with that exact name.
3. **Place the Model File:** Copy your trained model file (`.pth` or `.pt`) into the corresponding subdirectory.

**Example:** To add a model named `resnet_v3.pth` that predicts `Final Gravity`, the file must be placed at:
`models/Final Gravity/resnet_v3.pth`.

### Creating a Standalone Installer

The application is built using **PyInstaller** with a **`.spec` file**, which is the most robust method for handling
this project's complex dependencies.

1. **Install/Update Dependencies:** Make sure `pyinstaller` and `scikit-learn` are installed in your virtual
   environment.
   `pip install -U pyinstaller scikit-learn`

2. **Ensure `sys.path` Fix:** Confirm the path correction code is at the top of `app.py`.

3. **Ensure Hook File Exists:** Confirm the `hooks/hook-gradio.py` file exists with the correct content.

4. **Build from the Spec File:** The recommended way to build is to use the included `app.spec` file. This file
   contains all the necessary configurations to correctly bundle Gradio and its hidden dependencies.

    * First, delete any old `build` and `dist` folders to ensure a clean build.
    * Then, run the following command in your terminal:
      ```bash
      pyinstaller app.py.spec
      ```

5. **Distribute:** Find the final application folder inside `dist/BrewAI`. Zip this entire folder and distribute the ZIP
   file.

## 💻 Technology Stack

* **Python 3.10+**
* **Gradio:** For the user interface
* **PyTorch:** For loading and running the models
* **Pandas:** For data manipulation
* **scikit-learn:** A core dependency for the data processing models
* **PyInstaller:** For creating the standalone executable

## 📜 License

This project is licensed under the MIT License.
