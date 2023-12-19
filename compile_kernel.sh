
new_path="/usr/local/cuda/bin/"

# Check if the path exists in the PATH variable
if [[ ":$PATH:" != *":$new_path:"* ]]; then
    # Append the new path to the PATH variable
    export PATH="$PATH:$new_path"
    echo "Path added successfully."
else
    echo "Path already exists in the PATH variable."
fi

mkdir build
cd build
make
cp gra* ..
cd ..
