require(['c3', 'jquery', 'static/js/charts'], function(c3, $, charts) {

    var keywordStatsCharts = {};

    $( document ).ready(function() {
        $.ajax({
            type: "POST",
            url: "/keywords-top10-by-document-frequency",
            data: JSON.stringify({}),
            contentType: "application/json; charset=utf-8",
            dataType: "json",
            success: draw_top10,
            failure: function(errMsg) {
                console.log("failure");
                alert(errMsg);
            }
        });

        $( "body" ).on('click', "a.top10-keywords-link", function() {
            event.preventDefault();
            $('html,body').animate({ scrollTop: 0 }, 'slow');
            var keyword = $(this).text().trim();
            charts.addKeyword(keyword);

        })

    });

    function draw_top10(jsonData) {
        console.log(jsonData)

        Object.keys(jsonData['corpora']).forEach(function(corpus_name) {
            var corpus_div_id = "corpus-top10-" + corpus_name.replace(/_/g, '-').replace(/ /g, '-');

            var trs = ""
            for (var i=0; i < jsonData['corpora'][corpus_name].length; i++) {

                trs += `
                    <tr>
                      <td width="1"><h4> ${ i+1 } </h4></td>
                      <td><a href="#" class="top10-keywords-link"> ${ jsonData['corpora'][corpus_name][i]['keyword'] } </a></td>
                      <td class="text-right"><span class="text-muted"> ${ jsonData['corpora'][corpus_name][i]['df'] } </span></td>
                    </tr>
                `
            }

            $( "#row-top10-container" ).append(`
                <div class="col-lg-${Math.round(12.0 / Object.keys(jsonData['corpora']).length)}">
                    <div class="card">
                        <div class="card-header"><h3 class="card-title">${corpus_name.replace(/_/g, ' ').replace(/ /g, ' ')}</h3></div>
                        <table class="table card-table">
                            <tbody>
                                ${trs}
                            </tbody>
                        </table>
                    </div>
                </div>
            `);

            $( "table td" ).css("padding-top","3")
            $( "table td" ).css("padding-bottom", "3")

        });
    }
});
