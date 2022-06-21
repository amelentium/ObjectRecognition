function previewImages() {
    var preview = document.querySelector('#TrainImagesPreview');

    preview.innerHTML = ''

    if (this.files) {
        [].forEach.call(this.files, readAndPreview);
    }

    function readAndPreview(file) {
        if (!/\.(jpe?g|png)$/i.test(file.name)) {
            return alert(file.name + ' is not an image!')
        }
        var reader = new FileReader()
        reader.addEventListener('load', function () {
            var image = new Image()
            image.className = 'img-thumbnail'
            image.height = 150
            image.width = 150
            image.title = file.name
            image.src = this.result
            preview.appendChild(image)
        });
        reader.readAsDataURL(file)
    }
}
document.querySelector('#TrainImages').addEventListener('change', previewImages)
