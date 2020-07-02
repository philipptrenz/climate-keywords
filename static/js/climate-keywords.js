require(['c3', 'jquery'], function(c3, $) {

    var colorPattern = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'];
    var tagButton = '<button class="tag-button"><i class="fe fe-x"></i></button>';

    $( ".tag" ).append( tagButton );

    $('body').on('click', '.tag-button', function() {
        $( this ).parent().remove();
    });

    // process search bar submit
    $( "#keyword-search-form" ).submit(function( event ) {
        var value = $( ".keyword-search" ).val();
        $( ".keyword-search" ).val("");
        //$( "#card-filter" ).append('<span class="tag" style="background-color: ' + colorPattern[$( "#card-filter" ).length + 1] + ';">' + value + tagButton +'</span>');
        $( "#card-filter" ).append(`
            <span class="tag" style="background-color: ${ colorPattern[$( "#card-filter > span" ).length] };">
                ${ value }
                ${ tagButton }
            </span>
        `);

        event.preventDefault();
    });

    // gets triggered if tags get added or deleted
    $('body').on('DOMSubtreeModified', '#card-filter', function(){
        requestDataWithKeywords();
    });

    $( document ).ready(function() {
        requestDataWithKeywords();
    });

    function requestDataWithKeywords() {
        var keywords = [];
        $( "#card-filter > span" ).each(function() {
            keywords.push( $( this ).text() );
        });

        var start_time = new Date();
        $.ajax({
            type: "POST",
            url: "/data",
            data: JSON.stringify(keywords),
            contentType: "application/json; charset=utf-8",
            dataType: "json",
            success: function(data){
                var now = new Date();
                console.log('request took', Math.round((now-start_time)/10)/100, 's');
                drawChart(data);
            },
            failure: function(errMsg) {
                console.log("failure");
                alert(errMsg);
            }
        });
    }

    function drawChart(data){
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