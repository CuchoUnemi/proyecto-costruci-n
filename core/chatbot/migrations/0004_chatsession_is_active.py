# Generated by Django 4.2 on 2024-10-05 05:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0003_chatsession_session_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='chatsession',
            name='is_active',
            field=models.BooleanField(default=True),
        ),
    ]
