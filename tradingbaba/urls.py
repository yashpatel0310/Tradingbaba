
from django.contrib import admin
from django.urls import path, re_path
from django.conf.urls import url
from tradingbaba import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='home'),

]
