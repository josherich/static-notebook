MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
var converter = markdownit();
anchor(converter, {})

var GRAPH_WRITING_KEY = 'GRAPH_WRITING_CONTENT_TMP';
var SCALE_MAX = 2;
var SCALE_MIN = 0.5;
var content_url = './src/data.md'

function loadContent() {
  return window.localStorage.getItem(GRAPH_WRITING_KEY);
}

function saveContent() {
  var content = $('#content_editor').val();
  window.localStorage.setItem(GRAPH_WRITING_KEY, content);
}

function jump(h){
  document.getElementById(h).scrollIntoView();
}

var getGraphData = function(callback) {
  d3.text(content_url, function(text) {
    $('#content_editor').val(text)

    var tokens = converter.parse(text)

    var parsed = Dependent.parse(tokens)

    function linkIndexOf(id, nodes) {
      var index;
      nodes.map(function(n, i) {
        if (n == null) {
          index = 'NA'
        } else if (n.id === id) {
          index = i
        }
      });
      return index;
    }

    $(".content-body").html(converter.render(text))

    $('.content-body')
    .children()
    .each(function(e) {
      $(this).css({position: 'relative'})
      $(this).append($('<div class="marker">M</div>'))
    })

    var links = parsed.links.map(function(p) {
      return {
        source: linkIndexOf(p.source, parsed.nodes),
        target: linkIndexOf(p.target, parsed.nodes)
      }
    }).filter(function(p) {
      return p['source'] != undefined && p['target'] != undefined
    });

    // console.log(parsed.nodes)
    // console.log(parsed.links)
    // console.log(links)

    callback && callback(parsed.nodes, links)
  })
}


getGraphData(function(nodes, links) {

  var r = 10;
  var graph, zoom;
  var graphWidth, graphHeight;
  var history = [];
  var history_ptr = 0;
  var tree;
  var start;

  graphWidth = $('.graph').width() / 2;
  graphHeight = $('.graph').height();


  var getTreeData = function(source) {
    // var index = 0;

    // nodes.map(function(node, idx) {
    //   if (node == null) {
    //     index = 'NA'
    //   } else if (node.id === source.id) {
    //     index = idx
    //   }
    // });
    return buildTree(source);
  };

  var buildTree = function(source) {
    var treeObj = {};
    var children = [];

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

  function readNode(d) {
    Tree.render(getTreeData(d));
    jump(d.text);
  }

  function readPrevNode() {
    history_ptr -= 1
    if (history_ptr < 0) {
      history_ptr = 0
    } else {
      readNode(history[history_ptr])
    }
  }

  function readNextNode() {
    history_ptr += 1
    if (history_ptr > history.length - 1) {
      history_ptr = history.length - 1
    } else {
      readNode(history[history_ptr]);
    }
  }

  function onZoomChanged() {
    var scale = d3.event.scale
    // if (scale > SCALE_MAX) {
    //   scale = SCALE_MAX;
    // }
    // if (scale < SCALE_MIN) {
    //   scale = SCALE_MIN;
    // }
    graph.attr("transform", "translate(" + d3.event.translate + ")" + " scale(" + scale + ")")
  }

  function onControlZoomClicked(e) {
    var elmTarget = $(this)
    var scaleProcentile = 0.20;

    // Scale
    var currentScale = zoom.scale()
    var newScale
    if(elmTarget.hasClass('control-zoom-in')) {
      newScale = currentScale * (1 + scaleProcentile)
    } else {
      newScale = currentScale * (1 - scaleProcentile)
    }
    newScale = Math.max(newScale, 0)

    // Translate
    var centerTranslate = [
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
  $('.back').on('click', readPrevNode);
  $('.forward').on('click', readNextNode);
  zoom = d3.behavior.zoom();
  zoom.on("zoom", onZoomChanged);

  function render() {
    tree = Tree.render(getTreeData({"id":"machinelearning", text: "Machine Learning"}));
    start = {"id":"machinelearning", text: "Machine Learning"};

    graph = DAG.render({nodes: nodes, links: links}, zoom, function(d) {
      if (d['id'] !== history[history.length - 1]['id']) {
        history.push(d);
        history_ptr += 1;
      }
      readNode(d);
    });

    readNode(start);
    history = [];
    history.push(start);
  }

  function toggle_graph() {
    $('#graph').toggle()
  }
  function toggle_tree() {
    $('#tree').toggle()
  }
  function toggle_edit() {
    $('#editor').toggle()
  }

  function attachToStash(mark) {
    $('#marker_stash').append(mark)
  }
  // setTimeout(function() {
  //   DAG.focus(start);
  // }, 2000)

  render();

  $('#rerender').on('click', function(e) {
    saveContent();
    render();
    MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
  })

  $('.content').on('click', function(e) {
    if (e.target.nodeName !== 'A') return;
    var id = e.target.hash.replace('#', '');
    if (nodes.filter(function(n) { return n["id"] == id }).length > 0) {
      if (id !== history[history.length - 1]['id']) {
        history.push({"id": id});
        history_ptr += 1;
      }
      readNode({"id": id});
      console.log(history);
      console.log(history_ptr);
    }
  })

  $('#control_panel').on('click', function(e) {
    if (e.target.id === 'toggle_graph') {
      toggle_graph();
    } else if (e.target.id === 'toggle_tree') {
      toggle_tree();
    } else if (e.target.id === 'toggle_edit') {
      toggle_edit();
    }
  })

  $('.content-body').on('click', function(e) {
    if (e.target.className === 'marker') {
      var mark = $(e.target).parent().clone()
      var close = $(mark).find('.marker')
      $(close).text('x')
      attachToStash(mark)
    }
  })

  $('#marker_stash').on('click', function(e) {
    if (e.target.className === 'marker') {
      $(e.target).parent().remove();
    }
  })
})
