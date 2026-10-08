# Make the Streamlit site available without keeping Terminal open

A Streamlit app running on `localhost` only exists while the Python process is running. Closing Terminal stops that local server.

To get a permanent web address without leaving your Mac/Terminal running, deploy the project to a cloud host. The simplest workflow for this project is Streamlit Community Cloud.

## Streamlit Community Cloud workflow

1. Create a GitHub repository.
2. Upload the **contents of this project folder** to the repository. `app.py` and `requirements.txt` should be at the repository root.
3. Sign in to Streamlit Community Cloud with GitHub.
4. Create a new app and select that repository.
5. Choose `app.py` as the entry file.
6. Deploy.
7. Streamlit installs `requirements.txt` and gives the app a normal web URL.

After deployment, your Mac does not need to keep Terminal open. The remote server runs the application.

## Important hosting note

The application uses `yfinance`, so the cloud server needs outbound internet access. The bundled sector-universe CSV is stored inside the repository and does not depend on Wikipedia or another universe website.

## Updating the hosted site

Edit the GitHub repository (or push a new version). The hosted app can then rebuild from the updated code. Keep `config.yaml`, `data/us_large_cap_universe.csv`, and the `src/` folder in the repository.

## Local alternative

You can create a double-click launcher on a Mac, but that still runs a local Python server in the background. A genuinely independent web URL requires hosting/deployment.
