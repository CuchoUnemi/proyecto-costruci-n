# Generated by Django 4.2 on 2024-09-28 05:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0002_chatsession_message_delete_chathistory'),
    ]

    operations = [
        migrations.AddField(
            model_name='chatsession',
            name='session_name',
            field=models.CharField(blank=True, max_length=255),
        ),
    ]