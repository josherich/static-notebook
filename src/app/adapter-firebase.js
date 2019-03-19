let sync = {
  storageRef: null
}

sync.config = function(opt) {
  sync.storageRef = opt.storageRef
}

sync.get = function(file, resolve) {
  let fileRef = sync.storageRef.child(file)
  // Get the download URL
  return fileRef.getDownloadURL().then(function(url) {
      return fetch(url).then(response => response.text())
    }).catch(function(error) {
      switch (error.code) {
        case 'storage/object-not-found':
          console.log("File doesn't exist")
          break

        case 'storage/unauthorized':
          console.log("User doesn't have permission to access the object")
          break

        case 'storage/canceled':
          console.log("User canceled the upload")
          break

        case 'storage/unknown':
          console.log("Unknown error occurred, inspect the server response")
          break
      }
    })
}

sync.set = function(string, file) {
  let fileRef = sync.storageRef.child(file)
  fileRef.putString(string).then(function() {
    console.log('Uploaded')
  })
}

export default sync
