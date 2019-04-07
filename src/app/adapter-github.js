let sync = {}

sync.get = function(file, resolve) {
  let uri = file
  return fetch(uri).then(response => response.text())
}

sync.set = function(string, file) {
  // commit and push to file branch
}

export default sync
