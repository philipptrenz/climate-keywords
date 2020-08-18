require(['c3', 'jquery'], function(c3, $) {

    var keywordStatsCharts = {};

    $( document ).ready(function() {
        $.ajax({
            type: "POST",
            url: "/keywords-grouped-by-document-frequency",
            data: JSON.stringify({}),
            contentType: "application/json; charset=utf-8",
            dataType: "json",
            success: draw_stats,
            failure: function(errMsg) {
                console.log("failure");
                alert(errMsg);
            }
        });
    });


    function draw_stats(jsonData) {

        console.log(jsonData)

        Object.keys(jsonData['corpora']).forEach(function(corpus_name) {
            var corpus_div_id = "corpus-stats-" + corpus_name.replace(/_/g, '-').replace(/ /g, '-');

            $( "#row-stats-container" ).append(`
                <div class="col-lg-${Math.round(12.0 / Object.keys(jsonData['corpora']).length)}">
                    <div class="card">
                        <div class="card-header"><h3 class="card-title">${corpus_name.replace(/_/g, ' ').replace(/ /g, ' ')}</h3></div>
                        <div class="card-body" id="${corpus_div_id}" style="height: 15rem; padding: 0;"></div>
                    </div>
                </div>
            `)

            keywordStatsCharts[corpus_name] = c3.generate({
                bindto: '#' + corpus_div_id, // id of chart wrapper
                data: {
                    x: 'x',
                    columns: [
                        [ 'x' ].concat(jsonData['labels']),
                        [ 'df' ].concat(jsonData['corpora'][corpus_name])
                    ],
                    type: 'bar'
                },
                axis: {
                    y: {
                        label: '# keywords',
                        show: true,
                        tick: {
                            format: d3.format("s")
                        }
                    },
                    x: {
                        //label: 'keywords appear ...',
                        type: 'category',
                        show: true,
                        tick: {
                            multiline: false
                        },
                        height: 50,
                    }
                },
                legend: {
                    show: false,
                    position: 'inset',
                    padding: 0,
                },
                padding: {
                    left: 50,
                    top: 20,
                    right: 20,
                    bottom: 0
                },
                point: {
                    show: false
                }
            });

        });
    }
});
