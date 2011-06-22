function r(f){/loaded|complete/.test(document.readyState)?f():setTimeout("r("+f+")",9);}
function go() {
    var body = document.getElementsByTagName('body')[0];
    var e = document.createElement('p');
    e.setAttribute('class', 'cop');
    e.innerHTML =
    '<strong>Oxford Brookes University | A context-aware web recommendation system</strong> | '
    + '<a href="http://www.notmyidea.org/">Alexis Métaireau</a>'
    body.appendChild(e);
}
r(go);
