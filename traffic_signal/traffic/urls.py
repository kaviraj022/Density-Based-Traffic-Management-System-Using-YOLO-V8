from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_view, name='upload'),
    path('intersection/', views.intersection_view, name='intersection'),
] 