let sync = {}
const REPO = 'static-notebook'
sync.get = function(filePath, resolve) {
  let uri = filePath
  return fetch(uri).then(response => response.text())
}

sync.set = function(string, file) {
  // commit and push to file branch
}

export default sync
