function setUISignin() {
  document.querySelector('#firebaseui-auth-container').style.display = 'none'
  document.querySelector('#save').style.display = 'block'
  document.querySelector('#signin').style.display = 'none'
}
function setUISignout() {
  document.querySelector('#firebaseui-auth-container').style.display = 'none'
  document.querySelector('#save').style.display = 'none'
  document.querySelector('#signin').style.display = 'block'
}
// Initialize Firebase
let config = {
  apiKey: "AIzaSyDtAy_fTes6za3eiHXS6Ug9Fl0QVzLolgM",
  authDomain: "notebook-47f55.firebaseapp.com",
  projectId: "notebook-47f55",
  storageBucket: "notebook-47f55.appspot.com",
  messagingSenderId: "634188757877"
}
firebase.initializeApp(config)
let storage = firebase.storage()
let storageRef = storage.ref()
let ui = new firebaseui.auth.AuthUI(firebase.auth());
let uiConfig = {
  callbacks: {
    signInSuccessWithAuthResult: function(authResult, redirectUrl) {
      // User successfully signed in.
      // Return type determines whether we continue the redirect automatically
      // or whether we leave that to developer to handle.
      console.log(authResult)
      setUISignin()
      return true;
    },
    uiShown: function() {
      // The widget is rendered.
      // Hide the loader.
      // document.getElementById('loader').style.display = 'none';
    }
  },
  // Will use popup for IDP Providers sign-in flow instead of the default, redirect.
  signInFlow: 'popup',
  signInSuccessUrl: 'http://localhost:8081',
  signInOptions: [
    // Leave the lines as is for the providers you want to offer your users.
    firebase.auth.GoogleAuthProvider.PROVIDER_ID,
    firebase.auth.EmailAuthProvider.PROVIDER_ID
  ]
};

firebase.auth().onAuthStateChanged(function(user) {
  if (user) {
    setUISignin()
  } else {
    setUISignout()
  }
})

$('#signin').click((e) => {
  document.querySelector('#firebaseui-auth-container').style.display = 'block'
  ui.start('#firebaseui-auth-container', uiConfig);
})

$('#auth_close button').click((e) => {
  document.querySelector('#firebaseui-auth-container').style.display = 'none'
})