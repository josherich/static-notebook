(function(){
  var Dependent = {};
  Dependent.parse = function(tokens) {
    var blocks = [];
    var links = [];
    var token_len = tokens.length;
    var block = null;
    var token = null;

    function findLinks(inlines) {
      var inlines_len = inlines.length;
      var token = null;
      var _links = [];
      for (var i = 0; i < inlines_len; ) {
        token = inlines[i];
        if (token.type === 'link_open' && token.tag === 'a') {
          _links.push({text: inlines[i+1]['content'], target: token.attrs[0][1].replace('#', '')})
          i += 2;
        } else {
          i++;
        }
      }
      return _links;
    }

    function findStrong(inlines) {
      var inlines_len = inlines.length;
      var token = null;
      var _links = [];
      for (var i = 0; i < inlines_len; ) {
        token = inlines[i];
        if (token.type === 'strong_open') {
          _links.push({text: inlines[i+1]['content'], target: inlines[i+1]['content'].toLowerCase().replace(/\s/, '')})
          i += 2;
        } else {
          i++;
        }
      }
      return _links;
    }

    for (var i = 0; i < token_len; ) {
      token = tokens[i];
      var _children = [];
      if (token.type === 'heading_open' && token.tag[0] === 'h') {
        _children = findStrong(tokens[i+1].children);
        if (_children.length > 0) {
          block = {id: _children[0]['target'], text: _children[0]['text']}
        } else {
          // block = {id: tokens[i+1].content, text: tokens[i+1].content}
        }
        blocks.push(block)
        i += 2
      } else if (token.type === 'inline') {
        _children = findLinks(token.children);
        _children.map(function(c) {
          if (block) {
            links.push({source: block['id'], text: c['text'], target: c['target'].replace('#', '')})
          }
        });
        i++;
      } else {
        i++;
      }
    }

    return {
      nodes: blocks,
      links: links
    }
  }
  window.Dependent = Dependent;
})()