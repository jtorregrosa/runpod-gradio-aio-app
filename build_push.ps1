# Define variables
$repository = "jtorregrosa/runpod-gradio-aio-app"
$uniqueTag = "$(Get-Date -Format "yyyyMMdd-HHmmss")"
$latestTag = "latest"

# Docker image tag
$imageNameUnique = "${repository}:${uniqueTag}"
$imageNameLatest = "${repository}:${latestTag}"

# Build Docker image with best practices
Write-Output "Building Docker image with tag ${uniqueTag}..."
docker build -t ${imageNameUnique} .

# Add "latest" tag to the image
Write-Output "Tagging Docker image as latest..."
docker tag ${imageNameUnique} ${imageNameLatest}

# Push both unique and latest tags to Docker Hub
Write-Output "Pushing Docker image with unique tag to Docker Hub..."
docker push ${imageNameUnique}

Write-Output "Pushing Docker image with latest tag to Docker Hub..."
docker push ${imageNameLatest}

Write-Output "Docker image has been successfully built and pushed to Docker Hub with tags: ${uniqueTag} and latest"
