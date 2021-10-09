var VueMasonryPlugin = window["vue-masonry-plugin"].VueMasonryPlugin;
var saved_searchQuery = ""

function save_query(input_string) {

	saved_searchQuery = input_string
	return saved_searchQuery


}



function keywordify(input_string)  {

	var $commonWords = ['do','did','does','i','a','about','an','and','are','as','at','be','by','com','de','en','for','from','how','in','is','it','la','of','on','or','that','the','this','to','was','what','when','where','who','will','with','und','the','www','has','have'];
	//var $text = "This is some text. This is some text. Vending Machines are great.";
	var $text = input_string;
	var query_keywords = "";

	// Convert to lowercase
	$text = $text.toLowerCase();

	// replace unnesessary chars. leave only chars, numbers and space
	$text = $text.replace(/[^\w\d ]/g, '');

	var result = $text.split(' ');

	// remove $commonWords
	result = result.filter(function (word) {
	    return $commonWords.indexOf(word) === -1;
	});



	for (var member in result){
		query_keywords = query_keywords + " "
		query_keywords = query_keywords + result[member] 
	}

	//var query_keywords = result.append()

	// Unique words
	//result = result.unique();
	
	console.log(query_keywords);

	return query_keywords;

	

}

Vue.use(VueMasonryPlugin);

const vm = new Vue({
    el: '#jina-ui',
    data: {
        serverUrl: 'http://localhost:45678/search',
        top_k: 5,
        topkDocs: [],
        topkDocsDict: {},
        results: [],
        searchQuery: "",
        queryChunks: [],
        selectQueryChunks: [],
        queryItem: [],
        docItem: null,
        loadedItem: 0,
        loadedQuery: 0,
        searchQueryIsDirty: false,
        isCalculating: false,
        distThreshold: 999,
        sliderOptions: {
            dotSize: 14,
            width: 'auto',
            height: 4,
            contained: false,
            direction: 'ltr',
            data: null,
            min: 999,
            max: 0,
            interval: 0.01,
            disabled: false,
            clickable: true,
            duration: 0.5,
            adsorb: false,
            lazy: false,
            tooltip: 'active',
            tooltipPlacement: 'top',
            tooltipFormatter: void 0,
            useKeyboard: false,
            keydownHook: null,
            dragOnClick: false,
            enableCross: true,
            fixed: false,
            minRange: void 0,
            maxRange: void 0,
            order: true,
            marks: false,
            dotOptions: void 0,
            process: true,
            dotStyle: void 0,
            railStyle: void 0,
            processStyle: void 0,
            tooltipStyle: void 0,
            stepStyle: void 0,
            stepActiveStyle: void 0,
            labelStyle: void 0,
            labelActiveStyle: void 0,
        }
    },
    mounted: function () {

    },
    components: {
        'vueSlider': window['vue-slider-component'],
    },
    computed: {
        searchIndicator: function () {
            if (this.isCalculating) {
                return '⟳ Fetching new results...'
            } else if (this.searchQueryIsDirty) {
                return '... Typing'
            } else {

                return '✓ Done'
            }
        },
	uploadIndicator: function () {
            if (this.isCalculating) {
                return '⟳ Downloading Notes...'
            } else if (this.searchQueryIsDirty) {
                return '... Typing'
            } else {

                return '✓ Done'
            }
        }
    },
    watch: {
        searchQuery: function () {
            this.searchQueryIsDirty = true
            this.expensiveOperation()
        },
        distThreshold: function () {
            this.refreshAllCards();
        }
    },
    methods: {
        clearAllSelect: function () {
            vm.queryChunks.forEach(function (item, i) {
                item['isSelect'] = !item['isSelect'];
                vm.refreshAllCards();
            });
        },
        selectChunk: function (item) {
            item['isSelect'] = !item['isSelect'];
            vm.refreshAllCards();
        },
        refreshAllCards: function () {
            vm.topkDocsDict = new Map(vm.topkDocs.map(i => [i.id, {
                'text': i.text,
                'hlchunk': [],
                'renderHTML': i.text
            }]));

	    //vm.topkDocs.forEach(function (item, index){ 
	    //	console.log(vm.topkDocsDict.get(item['id'])['renderHTML'])
	    //});
	   
            vm.queryChunks.forEach(function (item, i) {
                if (!('isSelect' in item)) {
                    item['isSelect'] = true;
                }
                if ((item['isSelect'] ) ) {
                    item.matches.forEach(function (r) {
                        if (vm.topkDocsDict.has(r.parentId)) {
                            let dist = r.scores['cosine'].value
                           	//if ((dist < vm.distThreshold) )  {
				/*
			      	if ((dist > 0.15) )  {

                                console.log(1 - r.scores["cosine"].value.toFixed(3))
				vm.topkDocs = []
                                //vm.topkDocsDict.get(r.parentId)['hlchunk'].push({
                                //    'range': r.location,
                                //    'idx': i,
                                //    'dist': dist,
                                //    'range_str': r.location[0] + ',' + r.location[1]

                                }
				*/
				if ((dist < 0.15) )  {

                                console.log(1 - r.scores["cosine"].value.toFixed(3))
                                vm.topkDocsDict.get(r.parentId)['hlchunk'].push({
                                    'range': r.location,
                                    'idx': i,
                                    'dist': dist,
                                    'range_str': r.location[0] + ',' + r.location[1]
                                });

                            }
                            if (dist < vm.sliderOptions.min) {
                                vm.sliderOptions.min = dist.toFixed(2)
                            }
                            if (dist > vm.sliderOptions.max) {
                                vm.sliderOptions.max = dist.toFixed(2)
                            }

                        } else {
                            console.error(r.id);
                        }
                    });
                }
            });
            vm.topkDocsDict.forEach(function (value, key, map) {
                vm.topkDocsDict.get(key)['hlchunk'].sort(function (a, b) {
                    return b['range'][0] - a['range'][0]
                })
                var replace_map = new Map();
                value['hlchunk'].forEach(function (item) {
                    if (!replace_map.has(item['range_str'])) {
                        replace_map.set(item['range_str'], [])
                    }
                    replace_map.get(item['range_str']).push(item)

                })

                replace_map.forEach(function (item, kk, mm) {
                    value['renderHTML'] = replaceRange(value['renderHTML'], item[0]['range'][0], item[0]['range'][1], item)
                })
            })
            vm.$nextTick(function () {
                vm.$redrawVueMasonry('my-masonry');
            })
        },
        // This is where the debounce actually belongs.
        expensiveOperation: _.debounce(function () {
            this.isCalculating = true
            vm.selectQueryChunks.length = 0;
            $.ajax({
                url: this.serverUrl,
                type: "POST",
                contentType: "application/json",
                cache: false,
                data: JSON.stringify({
                    //"parameters": {"top_k": this.top_k},
		    //"parameters": {"distThreshold": 0.05},
                    //"data": [this.searchQuery.toLowerCase()]
		    //"data": ["Hello"]
		    "data" : [keywordify(save_query(this.searchQuery))]
                }),
                error: function (jqXHR, textStatus, errorThrown) {
                    console.log(jqXHR);
                    console.log(textSta0000);
                    console.log(errorThrown);
                },
                success: function (data) {
                    vm.topkDocs = data.data.docs[0].matches;
                    console.log('Number parents: ' + vm.topkDocs.length);
		    vm.topkDocsDict = new Map(vm.topkDocs.map(i => [i.id, {
                	'text': i.text,
                	'hlchunk': [],
                	'renderHTML': i.text
            	     }]));

		    doc_results = saved_searchQuery + "\n";


	    	    vm.topkDocs.forEach(function (item, index){ 
		    	//console.log(vm.topkDocsDict.get(item['id'])['renderHTML'])
		    	doc_results = doc_results + vm.topkDocsDict.get(item['id'])['renderHTML']
		    	console.log(doc_results)

	    	    });

		 
	   	    if(saved_searchQuery.includes("?")){	

	                $.ajax({
             			url: 'http://localhost:40000/search',
             			type: "POST",
             			contentType: "application/json",
             			cache: false,
             			data: JSON.stringify({
            				//"parameters": {"top_k": this.top_k},
	    				//"parameters": {"distThreshold": 0.05},
            				//"data": [this.searchQuery.toLowerCase()]
	    				//"data": ["Hello"]
	    			"data" : [doc_results]
	     	      		})
		    
                     	});

		     };

                    vm.queryChunks = data.data.docs[0].chunks;
                    console.log('Number chunks: ' + vm.queryChunks.length);
                    vm.refreshAllCards();
                    console.log('Success');
                },
                complete: function () {
                    vm.isCalculating = false
                    vm.searchQueryIsDirty = false
                    vm.$nextTick(function () {
                        vm.$redrawVueMasonry('my-masonry');
                    })
                }
            });
              

	    
	
	    
	    

        }, 5000)
    }
});

function replaceRange(s, start, end, chunks) {
    var content = s.substring(start, end)
    chunks.forEach(function (c) {
        content = "<span class=\"match-chunk query-chunk match-chunk-" + c.idx + "\" match-dist=" + c.dist + " style=\"background:" + selectColor(c.idx, true) + "\">" + content + "</span>"
    })
    return s.substring(0, start) + content + s.substring(end);
}

function selectColor(number, colored) {
    if (!colored) {
        return `#fff`;
    }
    const hue = number * 137.508; // use golden angle approximation
    return `hsl(${hue},50%,75%)`;
}

//const keyword_extractor = require("keyword_extractor")

//const sentence = "Hello and Welcome to the podcast"

//const extraction_result = 

/*keyword_extractor.extract(sentence,{
	language:"english",
	remove_digits: true,
	return_changed_case: true,
	remove_duplicates: false
});*/



