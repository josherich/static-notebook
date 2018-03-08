MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
var converter = markdownit();
var GRAPH_WRITING_KEY = 'GRAPH_WRITING_CONTENT_TMP';

// converter.block.ruler.before('reference', 'my_rule', function replace(state) {
//   console.log(state.tokens.map(function(e){return e.type}).join(":"));
// });
// converter.block.ruler.after('lheading', 'my_rule', function replace(state) {
//   console.log(state);
// });
// converter.block.ruler.after('reference', 'my_rule', function replace(state) {
//   console.log(state);
// });

function loadContent() {
  return window.localStorage.getItem(GRAPH_WRITING_KEY);
}

function saveContent() {
  var content = $('#content_editor').val();
  window.localStorage.setItem(GRAPH_WRITING_KEY, content);
}

var getGraphData = function() {
  var text = $('#content_editor').val()
  var local = loadContent();
  if (local) {
    text = local;
    $('#content_editor').val(local)
  }
  var tokens = converter.parse(text)
  var parsed = Dependent.parse(tokens);

  function linkIndexOf(id, nodes) {
    var index;
    nodes.map(function(n, i) {
      if (n.id === id) {
        index = i;
      }
    });
    return index;
  }

  $(".content-body").html(converter.render(text))

  var links = parsed.links.map(function(p) {
    return {
      source: linkIndexOf(p.source, parsed.nodes),
      target: linkIndexOf(p.target, parsed.nodes) || 0
    }
  });

  return {
    "directed": true,
    "multigraph": false,
    "graph":[],
    "nodes": parsed.nodes,
    "links": links
  };
};

var nodes = getGraphData()["nodes"];
var links = getGraphData()["links"];

var getTreeData = function(source) {
  var index = 0;
  nodes.map(function(node, idx) {
    if (node.id === source.id) {
      index = idx;
    }
  });
  return buildTree(index);
};

var buildTree = function(root_index) {
  var treeObj = {};
  var children = [];
  var node = nodes[root_index];

  if (!node) {
    throw Error("node index not exists.");
  }
  treeObj["name"] = node.id;
  treeObj["content"] = node.content;
  links.map(function(link, index) {
    if (link.source == root_index) {
      children.push(link.target);
    }
  });
  treeObj["children"] = children.map(buildTree);
  return treeObj;
};


(function(window) {

  var r = 10;
  var graph, zoom;
  var graphWidth, graphHeight;
  var history = [];
  var history_ptr = 0;
  var tree;
  var start;

  graphWidth = $('.graph').width() / 2;
  graphHeight = $('.graph').height();

  function readNode(d) {
    Tree.render(getTreeData(d));
    // $(`#${d.id} .markdown`).replaceWith(function() {
    //   var tokens = converter.parse($(this).text())
    //   Dependent.parse(tokens);
    //   return converter.render($(this).text());
    // });
    // $(".content-body").html($(`#${d.id}`).html());
    // TEX.render($(`#${d.id}`).html());
  }

  function readPrevNode() {
    history_ptr -= 1;
    if (history_ptr < 0) {
      history_ptr = 0;
    } else {
      readNode(history[history_ptr]);
    }
  }

  function readNextNode() {
    history_ptr += 1;
    if (history_ptr > history.length - 1) {
      history_ptr = history.length - 1;
    } else {
      readNode(history[history_ptr]);
    }
  }

  function onZoomChanged() {
    graph.attr("transform", "translate(" + d3.event.translate + ")" + " scale(" + d3.event.scale + ")");
  }

  function onControlZoomClicked(e) {
    var elmTarget = $(this)
    var scaleProcentile = 0.20;

    // Scale
    var currentScale = zoom.scale();
    var newScale;
    if(elmTarget.hasClass('control-zoom-in')) {
      newScale = currentScale * (1 + scaleProcentile);
    } else {
      newScale = currentScale * (1 - scaleProcentile);
    }
    newScale = Math.max(newScale, 0);

    // Translate
    var centerTranslate = [
      (graphWidth / 2) - (graphWidth * newScale / 2),
      (graphHeight / 2) - (graphHeight * newScale / 2)
    ];

    // Store values
    zoom
      .translate(centerTranslate)
      .scale(newScale);

    // Render transition
    graph.transition()
      .duration(500)
      .attr("transform", "translate(" + zoom.translate() + ")" + " scale(" + zoom.scale() + ")");

  }

  $('.control-zoom a').on('click', onControlZoomClicked);
  $('.back').on('click', readPrevNode);
  $('.forward').on('click', readNextNode);
  zoom = d3.behavior.zoom();
  zoom.on("zoom", onZoomChanged);

  function render() {  
    tree = Tree.render(getTreeData({"id":"Subgradient"}));
    start = {"id":"kernelfunction"};

    graph = DAG.render(getGraphData(), zoom, function(d) {
      history.push(d);
      history_ptr += 1;
      readNode(d);
    });

    readNode(start);
    history = [];
    history.push(start);
  }


  // setTimeout(function() {
  //   DAG.focus(start);
  // }, 2000)

  render();

  $('#rerender').on('click', function(e) {
    render();
    saveContent()
  });

  $('.content').on('click', function(e) {
    if (e.target.nodeName !== 'A') return;
    var id = e.target.hash.replace('#', '');
    if (nodes.filter(function(n) { return n["id"] == id }).length > 0) {
      history.push({"id": id});
      history_ptr += 1;
      readNode({"id": id});
      console.log(history);
      console.log(history_ptr);
    }
  });

})(window);