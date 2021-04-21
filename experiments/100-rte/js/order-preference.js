



function make_slides(f) {
  var   slides = {};

  slides.consent = slide({
     name : "consent",
     start: function() {
      exp.startT = Date.now();
      $("#consent_2").hide();
      exp.consent_position = 0;      
     },
    button : function() {
        exp.go(); //use exp.go() if and only if there is no "present" data.
    }
  });



  slides.i0 = slide({
     name : "i0",
     start: function() {
      exp.startT = Date.now();
     }
  });

  slides.instructions1 = slide({
    name : "instructions1",
    start: function() {
      $(".instruction_condition").html("Between subject intruction manipulation: "+ exp.instruction);
    }, 
    button : function() {
      exp.go(); //use exp.go() if and only if there is no "present" data.
    }
  });


  slides.instructions2 = slide({
    name : "instructions2",
    start: function() {
      $(".instruction_condition").html("Between subject intruction manipulation: "+ exp.instruction);
         $(".err").hide();
    }, 
    button : function() {
     var radio_button_list = document.getElementsByName('rating2');
     console.log(radio_button_list);
     var radio_button;
     var count;
	   rating = ""
     for(count = 0; count<radio_button_list.length; count++) {
	     if(radio_button_list[count].checked) {
		     rating = count+1;
	     }
     };
     if(rating == 2) { 
         exp.go(); //use exp.go() if and only if there is no "present" data.
     } else {
         $(".err").show();

     }
    }
  });



  slides.instructions3 = slide({
    name : "instructions3",
    start: function() { 
	    $(".err").hide();
    }, 
    button : function() {
       var radio_button_list = document.getElementsByName('rating3');
       console.log(radio_button_list);
       var radio_button;
       var count;
         rating = ""
       for(count = 0; count<radio_button_list.length; count++) {
           if(radio_button_list[count].checked) {
                   rating = count+1;
           }
       };
	    console.log(rating);
       if(rating == 1) { 
           exp.go(); //use exp.go() if and only if there is no "present" data.
       } else {
           $(".err").show();
    
       }
    }
  });

  slides.instructions5 = slide({
    name : "instructions5",
    start: function() {  }, 
    button : function() {
      exp.go(); //use exp.go() if and only if there is no "present" data.
    }
  });


  slides.instructions6 = slide({
    name : "instructions6",
    start: function() {  }, 
    button : function() {
      exp.go(); //use exp.go() if and only if there is no "present" data.
    }
  });


  slides.instructions7 = slide({
    name : "instructions7",
    start: function() {  }, 
    button : function() {
      exp.go(); //use exp.go() if and only if there is no "present" data.
    }
  });



  slides.instructions8 = slide({
    name : "instructions8",
    start: function() {  }, 
    button : function() {
      exp.go(); //use exp.go() if and only if there is no "present" data.
    }
  });




   var buttonFunction = function() {
       	//console.log(exp.sliderPost);
           //this.log_responses();
           _stream.apply(this); //use exp.go() if and only if there is no "present" data.
       }
   
   
   var learningPresentHandle = function(stim) {
         $(".err").hide();
         this.init_sliders();
         exp.sliderPost = null;
         this.stim = stim; //FRED: allows you to access stim in helpers
   
         $(".alien").html('<img id="pngFrame" src="images/'+stim.alien_image+'" style="height:130px;">');
   
         $(".object").html('<img id="pngFrame" src="images/'+stim.object_image+'" style="width:250px;">');

         article = "a"; //(["a","e","i","o","u"].includes(stim.adjective.substring(0,1))) ? "an" : "a"

   
         if(stim.negation == 1) {
           negation = "<b>not ";
         } else {
           negation = "<b>";
         }
         phrasing = negation+article+' '+stim.adjective+'</b>'
         if(stim.scalar_modifier != "SHOULD_NOT_BE_DISPLAYED") {
            phrasing = negation+article+''+['', '', '', ' very', 'n extremely'][stim.scalar_modifier-1]+' '+stim.adjective+'</b>'
         }

         $(".description").html('<p class=triangle-border left"> This is '+phrasing+' spaceship.</p>');
   
       }
   
   var init_slidersFunction_ctxt = function() {
         utils.make_slider("#slider0_ctxt", function(event, ui) {
           exp.sliderPost = ui.value;
         });
       }
   var init_slidersFunction = function() {
         utils.make_slider("#slider0", function(event, ui) {
           exp.sliderPost = ui.value;
         });
       }
   
   var present_handleSpeakerChoice = function(stim) {
         $("#choose_speaker_table").hide();
         $("#choice_request").hide()
         $("#choice_continue_1").hide()
         $("#choice_continue_2").hide()
   
         $("#speaker_choice_feedback").hide();
   
         
            $("#choose_speaker_table").show();
            $("#choice_request").show()
            $("#choice_continue_1").show()
            
         $(".err").hide();
         this.init_sliders();
         exp.sliderPost = null;
         this.stim = stim; //FRED: allows you to access stim in helpers
  
         $(".object").html('<img id="pngFrame" src="images/'+stim.alien_image+'" style="width:150px;">');

 
         var names_list = _.shuffle(names);
         $(".object1").html('<input type="image" onclick="_s.button_1(\'first\')" id="myimage_first" src="images/'+stim.object_image1+'" style="height:190px;"/>');
         $(".object2").html('<input type="image" onclick="_s.button_1(\'second\')" id="myimage_first" src="images/'+stim.object_image2+'" style="height:190px;"/>');

         if(stim.adj_index == 0) {
            modifier = "more "
         } else {
            modifier = ""
         }
         $(".description").html('<p class=triangle-border left"> Click on the <b>'+modifier+''+stim.adjective+'</b> spaceship.</p>');
         this.has_decided = false;
   
         for(w = 0; w < 2; w++) {
             document.getElementById("myimage_"+["first","second"][w]+"_out").style.border = "0px solid white";
         }
         this.response = undefined;
       }
   
   //console.log("MAYBE MAKE THE ADJECTIVE THING A GAME: PUT UP A FEW SPACESHIPS; AND PRODUCE A DESCRIPTION");
   
   
   var buttonSpeakerChoice = function(result) {
        if(this.response != undefined) {
           return 0;
        }
        this.response = result;
        $("#speaker_choice_feedback").show();

          document.getElementById("myimage_"+this.stim.correct_answer+"_out").style.border = "5px solid blue";


        if(this.stim.correct_answer == result) {
          document.getElementById("speaker_choice_feedback_p").style.color = "blue";
          feedback = "Correct!"

        } else {
          document.getElementById("speaker_choice_feedback_p").style.color = "red";
          feedback = "Incorrect!"
        }

       
        $("#speaker_choice_feedback").html(feedback);
        $("#choice_continue_1").hide()
        $("#choice_continue_2").show()

        this.log_responses();
    }

    var button2SpeakerChoice = function() {   
            _stream.apply(this); //use exp.go() if and only if there is no "present" data.
        }
    
    var init_slidersSpeakerChoice = function() {
          utils.make_slider("#slider0", function(event, ui) {
            exp.sliderPost = ui.value;
          });
        }
    
    
    
   
    log_responsesSpeakerChoice = function() {
        exp.data_trials.push({
          "quiz_response" : this.response,
          "correct_response" : this.stim.correct_answer,
          "object" : this.stim.object_image,
          "adjective" : this.stim.adjective,
          "slide_number" : exp.phase,
      //    "condition" : this.stim.condition,
//          "imgs" : this.stim.imgs,
  //        "item" : this.stim.item,
    //      "distractorValues" : this.stim.distractorValues
        });
    };







  slides.subjectivity = slide({
    name : "subjectivity",
    present : stimuliContext,
    start : function(stim) {
      console.log(stim);
      $(".err").hide();

      //exp.order_questionnaires = _.sample([[0,1],[1,0]])

      this.init_sliders();
      exp.sliderPost1 = null;
      exp.sliderPost2 = null;


    },

    button : function() {
    	//console.log(exp.sliderPost);
      result1 = document.getElementById("rofky_input").value;
      result2 = document.getElementById("glab_input").value;

      if (exp.sliderPost1 != null && exp.sliderPost2 != null) {
        this.log_responses();
        exp.go(); //use exp.go() if and only if there is no "present" data.
      } else {
        $(".err").show();
      }

    },

    log_responses : function() {
        //console.log(this.stim.condition);
        exp.data_trials.push({
          "adj1_subj" : (exp.order_questionnaires[0] == 0 ? exp.sliderPost1 : exp.sliderPost2),
          "adj2_subj" : (exp.order_questionnaires[0] == 0 ? exp.sliderPost2 : exp.sliderPost1),
          "order_questionnaires" : exp.order_questionnaires,
          "slide_number" : exp.phase
        });
    },


    init_sliders : function() {
      utils.make_slider("#slider_subj_1", function(event, ui) {
        exp.sliderPost1 = ui.value;
      });
      utils.make_slider("#slider_subj_2", function(event, ui) {
        exp.sliderPost2 = ui.value;
      });
    },
  });





  slides.multi_slider_context = slide({
    name : "multi_slider_context",
    present : explanations.concat(stimuliContext),
    present_handle : function(stim) {
      console.log(stim);
      $(".err").hide();
      $(".wrong").hide();

      //$(".err2").hide();
      this.init_sliders();
      exp.sliderPost = null;
      this.stim = stim; //FRED: allows you to access stim in helpers
	    console.log(stim);

	    console.log(stim.premise);
      $(".first-sentence").html(stim.premise);
      $(".second-sentence").html(stim.hypothesis);

      if(stim.explanation != null) {
         $(".explanation").html(stim.explanation);
      } else {
         $(".explanation").html("Does the first paragraph entail the second paragraph?");
      }

      console.log("DONE PRESENTING");
//      document.getElementById("completion").value = "";

     var radio_button_list = document.getElementsByName('response');
     var radio_button;
     var count;
     for(count = 0; count<radio_button_list.length; count++) {
       radio_button_list[count].checked = false;
     };    
     var radio_button_list = document.getElementsByName('rating');
     console.log(radio_button_list);
     var radio_button;
     var count;
     for(count = 0; count<radio_button_list.length; count++) {
       radio_button_list[count].checked = false;
     };
   


     document.getElementById("sentence_div").style.display = "block"; 
//     document.getElementById("question_div").style.display = "none"; 
   


    },

    button : function() {
         hasValue = false;
         var radio_button_list = document.getElementsByName('rating');
         console.log(radio_button_list);
         var radio_button;
         var count;
         for(count = 0; count<radio_button_list.length; count++) {
           if(radio_button_list[count].checked) {
               hasValue= true;
           }
         };
	    if(hasValue) {
               this.log_responses();
              _stream.apply(this); //use exp.go() if and only if there is no "present" data.
	    } else {
               $(".err").show();
	    }
    },

    init_sliders : function() {
      utils.make_slider("#slider0_ctxt", function(event, ui) {
        exp.sliderPost = ui.value;
      });
    },
    log_responses : function() {
        //console.log(this.stim.condition);
	   if(this.stim.question != null) {
             answer = document.querySelector('input[name="response"]:checked').value;
		   correct_answer = this.stim.answer;
	   } else {
             answer = "NA";
		   correct_answer = "NA";
	   }
        dataForThisTrial = {
          "premise" : this.stim.premise,
          "hypothesis" : this.stim.hypothesis,
          "model_rating" : this.stim.model_rating,
          "rating" : document.querySelector('input[name="rating"]:checked').value,
          "slide_number" : exp.phase
        };
        exp.data_trials.push(dataForThisTrial);
	    console.log(exp.data_trials[exp.data_trials.length-1]);

      dataExperiment= {
          "time_in_minutes" : (Date.now() - exp.startT)/60000,
	  "ProlificURL" : window.location.href
      };

xhr = new XMLHttpRequest();

serverIndex = _.sample([1,2,3,4,5,6,7,8,9,10], 1)

	    xhr.open('POST', 'https://stanford.edu/~mhahn2/cgi-bin/experiments/serverByTrial'+serverIndex+'/');

	    xhr.setRequestHeader('Content-Type', 'application/json');
	    jointData = JSON.stringify({"experiment" : dataExperiment, "trial" : dataForThisTrial});
	    console.log(jointData);

	    xhr.send(jointData);


    },
  });




  slides.subj_info =  slide({
    name : "subj_info",
    submit : function(e){
      //if (e.preventDefault) e.preventDefault(); // I don't know what this means.
      exp.subj_data = {
        language : $("#language").val(),
        enjoyment : $("#enjoyment").val(),
        asses : $('input[name="assess"]:checked').val(),
        age : $("#age").val(),
        gender : $("#gender").val(),
        education : $("#education").val(),
//        colorblind : $("#colorblind").val(),
        comments : $("#comments").val(),
        suggested_pay : $("#suggested_pay").val(),
        condition : exp.condition,
        context_first : exp.context_first,
        tutorial : exp.tutorial,
	      condition_index : exp.conditionIndex
      };
      exp.go(); //use exp.go() if and only if there is no "present" data.
    }
  });

  slides.thanks = slide({
    name : "thanks",
    start : function() {
      exp.data= {
        //  "trials" : exp.data_trials,
        //  "catch_trials" : exp.catch_trials,
          "system" : exp.system,
          //"condition" : exp.condition,
          "subject_information" : exp.subj_data,
          "time_in_minutes" : (Date.now() - exp.startT)/60000,
	//  "experiment": "forgetting-rating-49",
	  "ProlificURL" : window.location.href
      };

	    // TURK
//      setTimeout(function() {turk.submit(exp.data);}, 1000);


	    //      PROLIFIC
xhr = new XMLHttpRequest();

            serverIndex = _.sample([1,2,3,4,5,6,7,8,9,10])

	    xhr.open('POST', 'https://stanford.edu/~mhahn2/cgi-bin/experiments/serverByTrial'+serverIndex+'/');


	    // set `Content-Type` header
	     xhr.setRequestHeader('Content-Type', 'application/json');
	    //
	    // // send rquest with JSON payload
	     xhr.send(JSON.stringify(exp.data));
      $(".redirect_prolific").html("Please click on this link to record your participation: <br><br><b><a href='https://app.prolific.co/submissions/complete?cc=26E36E18'>Record Participation</a></b><br><br>If you do not do this, you will NOT GET PAID.");

    }
  });

  return slides;
}


function init() {
//   const jqueryScript = document.createElement('script')
//   jqueryScript.src = 'expt-files/stimuli1.js'
//   document.head.append(jqueryScript)
//   
//   const jqueryCheckInterval = setInterval(() => {
//     if (typeof window.jQuery !== 'undefined') {
//   	clearInterval(jqueryCheckInterval)
//   	// do something with jQuery here
//   	 console.log("TEST");
//	 initAfterGettingData();
//
//     }
//   }, 10)
//   //    Script.load("expt-files/stimuli1.js"); // includes code for myFancyMethod();
//   //    setStimuli(); // cool, no need for callbacks!
//   // https://stackoverflow.com/questions/21294/dynamically-load-a-javascript-file
//   //
//  

const jqueryScript = document.createElement('script')
	conditions = [];
	for(i = 0; i<60; i++) {
		conditions.push(i);
	}
	alreadyDone = []; //[1, 6, 8, 9, 10, 12, 15, 16, 17, 19, 20, 22, 25, 27, 30, 31, 32, 33, 35, 38, 39, 40, 41, 42, 43, 44, 45, 48, 50, 51, 52, 53, 55, 57, 59];
	while(1) {
           exp.conditionIndex = _.sample(conditions, 1)[0]; 
		failed=false;
	   for(j=0; j<alreadyDone.length; j++) {
		   if(alreadyDone[j] == exp.conditionIndex) {
			   failed=true;
			   break;
		   }
	   }
		if(!failed) {
			break;
		}
	}
                
        jqueryScript.src = '../../../block-certificates/process/output/process_RTE_PMLM_1billion_WithIndep.py_'+exp.conditionIndex+'.js'
        console.log( '../../../block-certificates/process/output/process_RTE_PMLM_1billion_WithIndep.py_'+exp.conditionIndex+'.js');
        jqueryScript.onload = () => {initAfterGettingData()}
        document.head.append(jqueryScript)
}


/// init ///
function initAfterGettingData() {

         stimuliContext = _.shuffle(stimuli.slice(0, 30)); //_.sample(stimuli, 30);
console.log(stimuliContext);

repeatWorker = false;
//  (function(){
//      var ut_id = "adj-order-preference";
//      if (UTWorkerLimitReached(ut_id)) {
//        $('.slide').empty();
//        repeatWorker = true;
//        alert("You have already completed the maximum number of HITs allowed by this requester. Please click 'Return HIT' to avoid any impact on your approval rating.");
//      }
//})();

  exp.current_score_click = 0;
  exp.total_quiz_trials_click = 0;

  exp.current_score = 0;
  exp.total_quiz_trials = 0;
  exp.hasDoneTutorialRevision = false;
  exp.shouldDoTutorialRevision = false;
  exp.hasEnteredInterativeQuiz = false;

  exp.trials = [];
  exp.catch_trials = [];
  exp.instruction = _.sample(["instruction1","instruction2"]);
  exp.system = {
      Browser : BrowserDetect.browser,
      OS : BrowserDetect.OS,
      screenH: screen.height,
      screenUH: exp.height,
      screenW: screen.width,
      screenUW: exp.width
    };
  //blocks of the experiment:
   exp.structure=[];
exp.structure.push('i0')
exp.structure.push('consent')
exp.structure.push( 'instructions1')
//exp.structure.push( 'instructions2')
//exp.structure.push( 'instructions3')
	//exp.structure.push( 'instructions4')

   exp.structure.push( 'multi_slider_context')

exp.structure.push( 'subj_info')
exp.structure.push( 'thanks');


  exp.data_trials = [];
  //make corresponding slides:
  exp.slides = make_slides(exp);

  exp.nQs = utils.get_exp_length(); //this does not work if there are stacks of stims (but does work for an experiment with this structure)
                    //relies on structure and slides being defined

  $('.slide').hide(); //hide everything

  //make sure turkers have accepted HIT (or you're not in mturk)
  $("#start_button").click(function() {
    if (turk.previewMode) {
      $("#mustaccept").show();
    } else {
      $("#start_button").click(function() {$("#mustaccept").show();});
      exp.go();
    }
  });

      exp.order_questionnaires = _.sample([[0,1],[1,0]])


  exp.go(); //show first slide
}
