import TurndownService from 'turndown'

let turndownService = new TurndownService()

turndownService = turndownService.addRule('mathmlinline', {
  filter: '.mjx-chtml.MathJax_CHTML',
  replacement: function (content, node, options) {
    return node.nextSibling.innerHTML
  }
})

turndownService = turndownService.addRule('mathmldisplay', {
  filter: '.mjx-chtml.MJXc-display',
  replacement: function (content, node, options) {
    return node.nextSibling.innerHTML
  }
})

let Service = {
  save: function(container) {
    let mds = []
    Array.prototype.slice.call(container.children).map((e) => {
      let md = turndownService.turndown(e.outerHTML)
      mds.push(md)
    })
    return mds.join('\n\n')
  }
}

export default Service