define(['c3', 'jquery'], function(c3, $) {

    var colorPattern = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'];
    var tagButton = '<button class="tag-button"><i class="fe fe-x"></i></button>';

    var jsonData = null;
    var jsonKeywords = [];
    var jsonKeywordsColorPalette = {};
    var charts = {};

    $( document ).ready(function() {
        $( ".tag" ).append( tagButton );
        requestDataWithKeywords([]);
    });

    // add keyword tag
    $( "#keyword-search-form" ).submit(function( event ) {
        var value = $( ".keyword-search" ).val();
        var keyword = value.trim().toLowerCase();
        addKeywordToCharts(keyword);

        $( ".keyword-search" ).val("");

        event.preventDefault();
    });

    function addKeywordToCharts(keyword) {
        if (!jsonKeywords.includes(keyword)) {
            jsonKeywords.push(keyword);

            //$( "#card-filter" ).append('<span class="tag" style="background-color: ' + colorPattern[$( "#card-filter" ).length + 1] + ';">' + value + tagButton +'</span>');
            $( "#card-filter" ).append(`
                <span class="tag" style="background-color: ${ colorPattern[$( "#card-filter > span" ).length] };">
                    ${ keyword }
                    ${ tagButton }
                </span>
            `);
            requestDataWithKeywords([keyword]);
        }
    }

    // delete keyword tag
    $('body').on('click', '.tag-button', function() {
        $( this ).parent().remove();
        var keyword = $( this ).parent().text().trim().toLowerCase();

        // remove keyword from jsonKeywords
        var index = jsonKeywords.indexOf(keyword);
        if (index !== -1) jsonKeywords.splice(index, 1);

        removeKeywordFromJson(keyword);
    });

    // recolor tags if element gets added or removed
    $('body').on('DOMSubtreeModified', '#card-filter', function(){
        $( "#card-filter > span" ).each(function(i) {
            $( this ).css("background-color", colorPattern[i] );
            jsonKeywordsColorPalette[$( this ).text().trim().toLowerCase()] = d3.rgb(colorPattern[i]);
        });
    });

    function removeKeywordFromJson(keyword) {
        if (jsonData === null) return;
        Object.keys(jsonData['corpora']).forEach(function(corpus_name) {

            charts[corpus_name].unload({
                ids: [keyword]
            });
            charts[corpus_name].data.colors(jsonKeywordsColorPalette);

            Object.keys(jsonData['corpora'][corpus_name]).forEach(function(data_type) {
                delete jsonData['corpora'][corpus_name][data_type][keyword];
            });
        });
    }

    function requestDataWithKeywords(keywordArray) {
        $.ajax({
            type: "POST",
            url: "/keywords-per-year",
            data: JSON.stringify(keywordArray),
            contentType: "application/json; charset=utf-8",
            dataType: "json",
            success: function(data){
                mergeNewDataIntoJsonData(data);
                drawChart();
            },
            failure: function(errMsg) {
                console.log("failure");
                alert(errMsg);
            }
        });
    }

    function mergeNewDataIntoJsonData(newData) {
        if (jsonData === null) {
            jsonData = newData;
        } else {
            Object.keys(newData['corpora']).forEach(function(corpus_name) {
                Object.keys(newData['corpora'][corpus_name]).forEach(function(data_type) {
                    Object.keys(newData['corpora'][corpus_name][data_type]).forEach(function(keyword) {

                        if (corpus_name in jsonData['corpora']) {
                            jsonData['corpora'][corpus_name][data_type][keyword] = newData['corpora'][corpus_name][data_type][keyword];
                        } else {
                            alert("Corpora are inconsistent between frontend and backend, please reload the page");
                        }

                    });
                });
            });
        }
    }

    var mouseOverX = -1;

    function drawChart(){
        Object.keys(jsonData['corpora']).forEach(function(corpus_name) {

            var chartContainerId = "chart-" + corpus_name.replace(/_/g, '-');
            var chartTile = `
                <div class="col-lg-12">
                    <div class="card">
                        <div class="card-header">
                            <h3 class="card-title">${ corpus_name.replace(/_/g, ' ') }</h3>
                        </div>
                        <div id="${ chartContainerId }" style="height: 15rem"></div>
                    </div>
                </div>
            `;
            if ( $("#" + chartContainerId).length == 0 ) {
                $( "#row-cards-container" ).append(chartTile);
            }

            var corpus = jsonData['corpora'][corpus_name];

            keys = []
            columns = []
            Object.keys(corpus).forEach(function(key) {
                keys.push(key)
                columns.push([key].concat(corpus[key]['norm'])) // start array with keyword for grouping
            });

            if (typeof charts[corpus_name] === 'undefined' || charts[corpus_name] == null) { // generate chart

                charts[corpus_name] = c3.generate({
                        bindto: '#' + chartContainerId, // id of chart wrapper
                        data: {
                                json: jsonData['corpora'][corpus_name]['norm'],/*,
                                keys: {
                                    // x: 'name', // it's possible to specify 'x' when category axis
                                    value: ['norm', 'tf', 'df'],
                                }*/
                                onmouseout: function(d) {
                                    if (mouseOverX != -1) {
                                        mouseOverX = -1;
                                        for (ch in charts) {
                                            charts[ch].tooltip.hide()
                                        }
                                    }
                                }
                        },
                        axis: {
                                y: {
                                    label: '% of documents p.a.',
                                    show: true,
                                    min: null,
                                },
                                x: {
                                    show: true,
                                    type: 'category',
                                    categories: jsonData['years'],
                                    tick: {
                                        multiline: false,
                                        culling: {
                                            max: 20
                                        }
                                    }
                                }
                        },
                        legend: {
                            show: false,
                            position: 'inset',
                            padding: 0,
                            inset: {
                                anchor: 'top-left',
                                x: 20,
                                y: 8,
                                step: 10
                            }
                        },
                        tooltip: {
                            format: {
                                title: function (x) {
                                    return '';
                                },
                                value: function (value, ratio, id) {
                                    var format = d3.format('.2%');
                                    return format(value);
                                }
                            },
                            position: function (data, width, height, element) {
                                var top =  15;
                                var left = parseInt(element.getAttribute('x')) + parseInt(element.getAttribute('width'));
                                var x = data[0].x;

                                if (mouseOverX != x) {
                                    mouseOverX = x;
                                    for (ch in charts) {
                                        if (charts[ch].internal.config.bindto != this.config.bindto) {
                                            charts[ch].tooltip.show({x: x})
                                        }
                                    }
                                }

                                return {top: top, left: left};
                            }
                        },
                        padding: {
                            top: 20,
                            bottom: 10,
                            left: 40,
                            right: 40
                        },
                        point: {
                            show: false
                        },
                        color: {
                            pattern: colorPattern
                        }
                });

            } else { // only reload data

                charts[corpus_name].load({
                    json: jsonData['corpora'][corpus_name]['norm']
                });
                charts[corpus_name].data.colors(jsonKeywordsColorPalette);
            }

        });
    }

    return {
        initialize: function() {
            requestDataWithKeywords([])
        },
        addKeyword: addKeywordToCharts
    }
});
