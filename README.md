# climate-keywords


## Install dependencies

```bash
# Install virtualenv to your local environment
pip3 install virtualenv

# Create virtual environment
virtualenv ctm

# Activate the virtual environment under Mac/Linux
source ctm/bin/activate 
# or under Windows
ctm\Scripts\activate

# Install dependencies to virtual environment
pip3 install -r requirements.txt
```

If you add more dependencies to this project, add them to
`requirements.txt` by using the following command and commit afterwards.

```bash
pip3 freeze > requirements.txt
```

## Configuring

This project provides a `default.config.json`, which defines several
parameters for data access and processing. You can change configurations
by copying this file and renaming it to `config.json`, which will be
preferred, but ignored by Git.

### Using Google Cloud Translate API

If you want to use the official Google Cloud Translate API, follow
[these steps](https://cloud.google.com/docs/authentication/end-user#creating_your_client_credentials)
to generate your `client_secrets.json` file, then add the path to this
file to `default.config.json` resp. `config.json` under *translator* >
*google_client_secret_file*.
