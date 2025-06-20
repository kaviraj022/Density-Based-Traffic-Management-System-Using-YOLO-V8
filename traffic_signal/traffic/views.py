from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os

LANES = ['north', 'south', 'east', 'west']

def upload_view(request):
    if request.method == 'POST':
        files = {}
        for lane in LANES:
            uploaded = request.FILES.get(lane)
            if uploaded:
                fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, lane))
                filename = fs.save(uploaded.name, uploaded)
                files[lane] = fs.url(filename)
        request.session['lane_files'] = files
        return redirect('upload')  # For now, just reload the page
    return render(request, 'traffic/upload.html')
