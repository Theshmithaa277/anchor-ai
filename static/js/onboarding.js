/*
 * onboarding.js — small shared behaviour for the auth screens.
 * Progressive enhancement only: login.html and signup.html work fine as
 * plain server-rendered forms without this file loading at all.
 */
(function () {
  document.querySelectorAll('[data-password-toggle]').forEach(function (btn) {
    var targetId = btn.getAttribute('data-password-toggle');
    var input = document.getElementById(targetId);
    if (!input) return;

    btn.addEventListener('click', function () {
      var showing = input.type === 'text';
      input.type = showing ? 'password' : 'text';
      btn.textContent = showing ? 'Show' : 'Hide';
      btn.setAttribute('aria-pressed', String(!showing));
    });
  });
})();
