import markdownit from 'markdown-it'
import anchor_plugin from './anchor'
import footnote_plugin from './footnote'

let content_cache = null
let converter = markdownit()
anchor_plugin(converter, {})
footnote_plugin(converter)

const slugify = anchor_plugin.defaults.slugify

let SemanticDocs = {
  sync: null
}

SemanticDocs.config = function(opt) {
  SemanticDocs.sync = opt.sync
}

SemanticDocs.slugify = slugify

SemanticDocs.parse = function(tokens, options) {
  let blocks = []
  let links = []
  let token_len = tokens.length
  let block = null
  let token = null
  let use_strong = options.use_strong

  function findLinks(inlines) {
    let inlines_len = inlines.length
    let token = null
    let _links = []
    for (let i = 0; i < inlines_len; ) {
      token = inlines[i]
      if (token.type === 'link_open' && token.tag === 'a') {
        _links.push({text: inlines[i+1]['content'], target: token.attrs[0][1].replace('#', '')})
        i += 2
      } else {
        i++
      }
    }
    return _links;
  }

  function findStrong(inlines) {
    let inlines_len = inlines.length;
    let token = null
    let _links = []
    for (let i = 0; i < inlines_len; ) {
      token = inlines[i]
      if (token.type === 'strong_open') {
        _links.push({
          target: slugify(inlines[i+1]['content'], {lower: true}),
          text: inlines[i+1]['content']
        })
        i += 2
      } else {
        i++
      }
    }
    return _links
  }

  function extractLink(parent, link) {
    let result = {
      source: block['id'],
      text: link['text'],
      target: link['target'],
      type: 'url'
    }
    if (link['target'][0] === '#') {
      result.type = 'hash'
      result.target = link['target'].replace(/^#/, '')
    } else {
      result.block = {
        id: link['target'],
        text: link['text'],
        url: link['target']
      }
    }
    return result
  }

  function linkIndexOf(id, nodes) {
    let index;
    nodes.map(function(n, i) {
      if (n == null) {
        index = 'NA'
      } else if (n.id === id) {
        index = i
      }
    });
    return index
  }

  for (let i = 0; i < token_len; ) {
    token = tokens[i]
    let _children = []
    if (token.type === 'heading_open' && token.tag[0] === 'h') {
      if (use_strong) {
        _children = findStrong(tokens[i+1].children);
        if (_children.length > 0) {
          block = {id: _children[0]['target'], text: _children[0]['text']}
        }
      } else {
        block = {id: slugify(tokens[i+1].content.replace(/\*\*/g, ''), {lower: true}), text: tokens[i+1].content.replace(/\*\*/g, '') }
      }
      blocks.push(block)
      i += 2
    } else if (token.type === 'inline') {
      _children = findLinks(token.children);
      _children.map(function(child) {
        if (block) {
          let link = extractLink(block, child)
          if (link.type === 'url') {
            blocks.push(link.block)
          }
          links.push(link)
          // links.push({source: block['id'], text: c['text'], target: c['target'].replace('#', '')})
        }
      })
      i++
    } else {
      i++
    }
  }

  links = links.map(function(p) {
    return {
      source: linkIndexOf(p.source, blocks),
      target: linkIndexOf(p.target, blocks)
    }
  }).filter(function(p) {
    return p['source'] != undefined && p['target'] != undefined
  })

  let index = tokens
    .filter((token, idx) => {
      return token.type === 'heading_open' || (token.type === 'inline' && tokens[idx - 1].type == 'heading_open')
    })
    .map(token => {
      return {
        type: token.type,
        tag: token.tag,
        content: token.content,
        attrs: token.attrs
      }
    })
  return {
    nodes: blocks,
    links: links,
    index: index
  }
}

SemanticDocs.data = (filepath, use_strong) => {
  function parse(text) {
    let tokens = converter.parse(text, {})

    let parsed = SemanticDocs.parse(tokens, {
      use_strong: use_strong
    })

    return {
      nodes: parsed.nodes,
      links: parsed.links,
      index: parsed.index,
      text: converter.render(text)
    }
  }

  if (content_cache) {
    return new Promise((resolve, reject) => {
      resolve(parse(content_cache))
    })
  } else {
    let tasks = []
    if (!Array.isArray(filepath)) {
      filepath = [filepath]
    }

    filepath.map(file => {
      tasks.push(SemanticDocs.sync.get(file))
    })

    return Promise.all(tasks)
      .then(textArray => {
        let alltext = textArray.join('\n')
        content_cache = alltext
        return parse(alltext)
      })
  }
}

export default SemanticDocs
