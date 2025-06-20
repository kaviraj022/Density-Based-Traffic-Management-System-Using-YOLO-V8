window.onload = function() {
    const lanes = ['north', 'south', 'east', 'west'];
    const currentGreen = document.querySelector('.signal.green').id.split('-')[0];
    lanes.forEach(lane => {
        const video = document.getElementById(lane + '-video');
        if (video) {
            if (lane === currentGreen) {
                video.play();
            } else {
                video.pause();
            }
        }
    });
}; 