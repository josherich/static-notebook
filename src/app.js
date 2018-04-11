MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});

import SemanticDocs from './index'
import DAG from './dag'

let GRAPH_WRITING_KEY = 'GRAPH_WRITING_CONTENT_TMP';
let GRAPH_WRITING_OPTION_STRONG = 'GRAPH_WRITING_OPTION_STRONG';
let SCALE_MAX = 2;
let SCALE_MIN = 0.5;
let node_history = [];
let history_ptr = 0;
let start = null;
let use_cache = true;

function loadContent() {
  return window.localStorage.getItem(GRAPH_WRITING_KEY);
}

function saveContent() {
  let content = $('#content_editor').val();
  window.localStorage.setItem(GRAPH_WRITING_KEY, content);
}

function jump(h){
  document.getElementById(h).scrollIntoView();
}

function getRenderStrongNode() {
  let strong = localStorage.getItem(GRAPH_WRITING_OPTION_STRONG)
  return !!strong
}

function setRenderStrongNode(use_strong) {
  if (use_strong) {
    localStorage.setItem(GRAPH_WRITING_OPTION_STRONG, 'true')
  } else {
    localStorage.removeItem(GRAPH_WRITING_OPTION_STRONG)
  }
}

const startup = function(filepath, cache=true) {
  use_cache = cache;

  SemanticDocs.data(filepath, getRenderStrongNode())
  .then(data => {
    let nodes = data.nodes
    let links = data.links
    let index = data.index
    let text = data.text

    let r = 10;
    let graph, zoom;
    let graphWidth, graphHeight;
    let tree;

    graphWidth = $('.graph').width();
    graphHeight = $('.graph').height();

    let buildTree = function(source) {
      let treeObj = {};
      let children = [];

      treeObj["name"] = source.id;
      treeObj["content"] = source.text;
      links.map(function(link, index) {
        if (link.source.id == source.id) {
          children.push(link.target);
        }
      });
      treeObj["children"] = children.map(buildTree);
      return treeObj;
    };

    let buildIndex = index => {
      let frag = document.createDocumentFragment()
      index.map((node, idx) => {
        if (node.type === 'heading_open') {
          frag.appendChild($(`<div class=${node.tag}><a href="#${node.attrs[0][1]}">${index[idx + 1].content}</a></div>`)[0])
        }
      })
      $('#index').append(frag)
    }

    function readNode(node) {
      // Tree.render(buildTree(node));
      jump(SemanticDocs.slugify(node.text, {lower: true}))
    }

    function readPrevNode() {
      history_ptr -= 1
      if (history_ptr < 0) {
        history_ptr = 0
      } else {
        readNode(node_history[history_ptr])
      }
    }

    function readNextNode() {
      history_ptr += 1
      if (history_ptr > node_history.length - 1) {
        history_ptr = node_history.length - 1
      } else {
        readNode(node_history[history_ptr]);
      }
    }

    function onZoomChanged() {
      let scale = d3.event.scale
      // if (scale > SCALE_MAX) {
      //   scale = SCALE_MAX;
      // }
      // if (scale < SCALE_MIN) {
      //   scale = SCALE_MIN;
      // }
      graph.attr("transform", "translate(" + d3.event.translate + ")" + " scale(" + scale + ")")
    }

    function onControlZoomClicked(e) {
      let elmTarget = $(this)
      let scaleProcentile = 0.50;

      // Scale
      let currentScale = zoom.scale()
      let newScale
      if(elmTarget.hasClass('control-zoom-in')) {
        newScale = currentScale * (1 + scaleProcentile)
      } else {
        newScale = currentScale * (1 - scaleProcentile)
      }
      newScale = Math.max(newScale, 0)

      // Translate
      graphWidth = $('.graph').width();
      graphHeight = $('.graph').height();

      let centerTranslate = [
        (graphWidth / 2) - (graphWidth * newScale / 2),
        (graphHeight / 2) - (graphHeight * newScale / 2)
      ];

      // Store values
      zoom
        .translate(centerTranslate)
        .scale(newScale)

      // Render transition
      graph.transition()
        .duration(500)
        .attr("transform", "translate(" + zoom.translate() + ")" + " scale(" + zoom.scale() + ")")
    }

    $('.control-zoom a').on('click', onControlZoomClicked);

    zoom = d3.behavior.zoom();
    zoom.on("zoom", onZoomChanged);

    function renderContent(text) {
      // $('#content_editor').val(text)
      $(".content-body").html(text)
      $('.content-body')
      .children()
      .each(function(e) {
        $(this).css({position: 'relative'})
        $(this).append($('<div class="marker"><svg class="icon icon-files-empty"><use xlink:href="#icon-files-empty"></use></svg></div>'))
      })
    }

    function renderGraph(nodes, links, start, zoom) {
      start = nodes[0]
      // tree = Tree.render(buildTree(start));
      graph = DAG.render({nodes: nodes, links: links}, zoom, function(d) {
        if (d['id'] !== node_history[node_history.length - 1]['id']) {
          node_history.push(d);
          history_ptr += 1;
        }
        readNode(d);
      });
      readNode(start);
      node_history = [];
      node_history.push(start);
    }

    // setTimeout(function() {
    //   DAG.focus(start);
    // }, 2000)

    renderContent(text)
    MathJax.Hub.Queue(["Typeset",MathJax.Hub]);

    renderGraph(nodes, links, start, zoom)

    buildIndex(index)

    function attachToStash(mark) {
      $('#marker_stash').append(mark)
    }

    function toggle_graph() {
      // $('#graph').toggle()
      $('body').removeClass('index_visible')
      $('body').toggleClass('graph_visible')
    }

    function toggle_setting() {
      $('.modal.js').toggle()
    }

    function toggle_tree() {
      $('#tree').toggle()
    }

    function toggle_index() {
      // $('#index').toggle()
      $('body').removeClass('graph_visible')
      $('body').toggleClass('index_visible')
    }

    function toggle_edit() {
      $('#editor').toggle()
    }

    $('#control_panel #toggle_graph').on('click', toggle_graph)
    $('#control_panel #toggle_index').on('click', toggle_index)
    $('#control_panel #content_back').on('click', readPrevNode)
    $('#control_panel #content_forward').on('click', readNextNode)
    $('#control_panel #toggle_setting').on('click', toggle_setting)

    $('#rerender').on('click', function(e) {
      saveContent();
      renderGraph();
      MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
    })

    $('.content').on('click', function(e) {
      if (e.target.nodeName !== 'A') return;
      let id = e.target.hash.replace('#', '');
      if (nodes.filter(function(n) { return n["id"] == id }).length > 0) {
        if (id !== node_history[node_history.length - 1]['id']) {
          node_history.push({"id": id});
          history_ptr += 1;
        }
        readNode({"id": id});
      }
    })

    $('.content-body').on('click', function(e) {
      if (e.target.parentNode.className === 'marker') {
        let mark = $(e.target).parent().parent().clone()
        let close = $(mark).find('.marker')
        $(close).text('x')
        attachToStash(mark)
      }
    })

    $('#marker_stash').on('click', function(e) {
      if (e.target.className === 'marker') {
        $(e.target).parent().remove();
      }
    })

    $('.modal.js .modal-close').on('click', function(e) {
      SemanticDocs.data(filepath, getRenderStrongNode())
      .then(data => {
        let nodes = data.nodes
        let links = data.links
        renderGraph(nodes, links, start, zoom)
      })

      toggle_setting();
      MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
    })

    $('#toggle_strong_node').on('click', function(e) {
      setRenderStrongNode(e.target.checked)
    })

    $('#toggle_strong_node').attr({ checked: getRenderStrongNode()})

  })
}

export default startup