require(['c3', 'jquery'], function(c3, $) {

    var colorPattern = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'];
    var tagButton = '<button class="tag-button"><i class="fe fe-x"></i></button>';

    var jsonData = null;

    $( ".tag" ).append( tagButton );

    // delete tag
    $('body').on('click', '.tag-button', function() {
        $( this ).parent().remove();
        var keyword = $( this ).parent().text().trim().toLowerCase();
        removeKeywordFromJson(keyword);
        drawChart();
    });

    // process search bar submit
    $( "#keyword-search-form" ).submit(function( event ) {
        var value = $( ".keyword-search" ).val();
        var keyword = value.trim().toLowerCase();
        $( ".keyword-search" ).val("");
        //$( "#card-filter" ).append('<span class="tag" style="background-color: ' + colorPattern[$( "#card-filter" ).length + 1] + ';">' + value + tagButton +'</span>');
        $( "#card-filter" ).append(`
            <span class="tag" style="background-color: ${ colorPattern[$( "#card-filter > span" ).length] };">
                ${ keyword }
                ${ tagButton }
            </span>
        `);
        requestDataWithKeywords([keyword]);
        event.preventDefault();
    });

    $( document ).ready(function() {
        requestDataWithKeywords([]);
    });

    // gets triggered if tags get added or deleted
    $('body').on('DOMSubtreeModified', '#card-filter', function(){
        recolorKeywordTags();
    });

    function recolorKeywordTags() {
        $( "#card-filter > span" ).each(function(i) {
            $( this ).css("background-color", colorPattern[i] );
        });
    }

    function removeKeywordFromJson(keyword) {
        if (jsonData === null) return;

        console.log("removing "+keyword+" from json data")
        Object.keys(jsonData['corpora']).forEach(function(corpus_name) {
            delete jsonData['corpora'][corpus_name][keyword];
        });
    }

    function requestDataWithKeywords(keywordArray) {
        $.ajax({
            type: "POST",
            url: "/data",
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
                Object.keys(newData['corpora'][corpus_name]).forEach(function(keyword) {
                    if (corpus_name in jsonData['corpora']) {
                        jsonData['corpora'][corpus_name][keyword] = newData['corpora'][corpus_name][keyword];
                    } else {
                        alert("Corpora are inconsistent between frontend and backend, please reload the page");
                    }
                });
            });
        }
    }

    function drawChart(){
        var data = jsonData;

        if (typeof data === 'undefined' || data == null) return;
        console.log(data)

        Object.keys(data['corpora']).forEach(function(corpus_name) {

            var chartContainerId = "chart-" + corpus_name.replace(/_/g, '-');
            var chartTile = `
                <div class="col-lg-12">
                    <div class="card">
                        <div class="card-header">
                            <h3 class="card-title">${ corpus_name.replace(/_/g, ' ') }</h3>
                        </div>
                        <div id="${ chartContainerId }" style="height: 10rem"></div>
                    </div>
                </div>
            `;
            if ( $("#" + chartContainerId).length == 0 ) {
                $( "#row-cards-container" ).append(chartTile);
            }

            var corpus = data['corpora'][corpus_name];

            keys = []
            columns = []
            Object.keys(corpus).forEach(function(key) {
                keys.push(key)
                columns.push([key].concat(corpus[key]['norm'])) // start array with keyword for grouping
            });

            var chart = c3.generate({
                    bindto: '#' + chartContainerId, // id of chart wrapper
                    data: {
                            columns: columns,
                            type: 'line',


                            /*
                            groups: [
                                    keys
                            ],
                            colors: {
                                    'data1': tabler.colors["blue"]
                            },
                            names: {
                                    // name of each serie
                                    'data1': 'Data type'
                            }
                            */
                    },
                    axis: {
                            y: {
                                    label: 'Percentage of documents per year',
                                    padding: {
                                        bottom: 0,
                                    },
                                    show: true,
                                    min: 0,
                                    type: 'category',
                                    tick: {
                                        multiline: false,
                                        values: [0, 0.1, 0.2, 0.3, 0.4]
                                    },

                                    /*
                                    max: 1.0,
                                    tick: {
                                            outer: false
                                    }*/
                            },
                            x: {
                                    padding: {
                                        left: 5,
                                        right: 5
                                    },
                                    show: true,
                                    type: 'category',
                                    categories: data['years'],
                                    tick: {
                                        multiline: false,
                                        culling: {
                                            max: 20
                                        }
                                    }
                            }
                    },
                    legend: {
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
                                    }
                            }
                    },
                    padding: {
                            bottom: 0,
                            left: -1,
                            right: -1
                    },
                    point: {
                            show: false
                    },
                    color: {
                        pattern: colorPattern
                    }
            });

        });
    }
});
