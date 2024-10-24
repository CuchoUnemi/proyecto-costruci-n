"""
Django settings for config project.

Generated by 'django-admin startproject' using Django 4.2.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-a5&bzm5+r&p)@dht@8cwx)r#n0v31@#any-#@*ad-t5kcyxstb'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Módulos específicos del proyecto
    'core.chatbot.apps.ChatbotConfig',
    'core.home.apps.HomeConfig',
    'core.login_register.apps.LoginRegisterConfig',
    'core.profiles.apps.ProfilesConfig',
    'core.recover_password.apps.RecoverPasswordConfig',
    'core.webscraping.apps.WebscrapingConfig',
    # Aplicaciones de terceros
    'tailwind',
    'theme',
]
# Configuración para la aplicación de Tailwind CSS
TAILWIND_APP_NAME = 'theme'


INTERNAL_IPS = [
    "127.0.0.1",
]

# Ruta al ejecutable de npm para gestionar dependencias frontend
NPM_BIN_PATH = 'C:/Program Files/nodejs/npm.cmd'

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# El tiempo de vida del link de recuperar contra
# PASSWORD_RESET_TIMEOUT = 300

# RESTABLECER CONTRASENIA (NO SE USA PERO BUE)
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'cuchostorevip@gmail.com'  # Tu dirección de correo electrónico
EMAIL_HOST_PASSWORD = 'apgk nrjh srfz deun'  # Tu la llave de seguridad del correo electrónico
DEFAULT_FROM_EMAIL = 'cuchostorevip@gmail.com'  # Direccion por defaul



BROWSER_REFRESH_SECONDS = 1  # indica que el navegador se recargará automáticamente cada 1 segundo cuando se detecten
# cambios.

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

DATABASES = {
    "default": {
        'ENGINE': os.environ.get("DB_ENGINE", "django.db.backends.postgresql"),
        'NAME': os.environ.get("DB_DATABASE", ""),
        'USER': os.environ.get("DB_USERNAME", ""),
        'PASSWORD': os.environ.get("DB_PASSWORD", ""),
        'HOST': os.environ.get("DB_SOCKET", ""),
        'PORT': os.environ.get("DB_PORT", "5432"),
        'ATOMIC_REQUESTS': True
    }
}
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': BASE_DIR / 'db.sqlite3',
#     }
# }

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'es-ec'

TIME_ZONE = 'America/Guayaquil'

USE_I18N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = 'static/'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),  # carpeta física de archivos statics
    os.path.join(BASE_DIR, 'node_modules'),
]



MEDIA_URL = '/media/'  # url de imagenes
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field


DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

AUTH_USER_MODEL = 'profiles.User'
LOGIN_URL = '/auth/login/'


# # logs para verificar las consultas
# LOGGING = {
#     'version': 1,
#     'disable_existing_loggers': False,
#     'handlers': {
#         'console': {
#             'level': 'DEBUG',
#             'class': 'logging.StreamHandler',
#         },
#     },
#     'loggers': {
#         'django.db.backends': {
#             'level': 'DEBUG',
#             'handlers': ['console'],
#         },
#     },
# # }
# import logging

# class CustomFormatter(logging.Formatter):
#     def format(self, record):
#         original_msg = super().format(record)
#         return f"\n{'#' * 10} INICIA LOGS {'#' * 10}\n{original_msg}\n{'#' * 10} FINALIZA LOGS {'#' * 10}"

# LOGGING = {
#     'version': 1,
#     'disable_existing_loggers': False,
#     'handlers': {
#         'console': {
#             'level': 'DEBUG',
#             'class': 'logging.StreamHandler',
#             'formatter': 'custom',
#         },
#     },
#     'formatters': {
#         'custom': {
#             '()': CustomFormatter,
#             'format': '%(message)s',
#         },
#     },
#     'loggers': {
#         'django.db.backends': {
#             'level': 'DEBUG',
#             'handlers': ['console'],
#         },
#     },
# }