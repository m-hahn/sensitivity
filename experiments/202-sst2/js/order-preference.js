



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

  slides.motivation = slide({
    name : "motivation",
    start: function() {
    }, 
    button : function() {
      exp.go(); //use exp.go() if and only if there is no "present" data.
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
    }, 
    button : function() {
      exp.go(); //use exp.go() if and only if there is no "present" data.
    }
  });



  slides.instructions3 = slide({
    name : "instructions3",
    start: function() { }, 
    button : function() {
      exp.go(); //use exp.go() if and only if there is no "present" data.
    }
  });

  slides.instructions4 = slide({
    name : "instructions4",
    start: function() { }, 
    button : function() {
      exp.go(); //use exp.go() if and only if there is no "present" data.
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


  slides.example1 = slide({
    name : "example1",
    start : function() {
      console.log("START EXP");
      $(".err").hide();

      this.review = ["I", "simply", "can't", "recommend", "this", "movie", "enough."]

      this.label = 1

      this.canChange = [];
      this.rows = [];
      aggregateLength = 0
	    row = 0;
      for(i=0; i<this.review.length; i++) {
	      aggregateLength+=this.review[i].length;
	      if(aggregateLength > 50) {
		      row += 1;
		      aggregateLength = this.review[i].length;
	      }
	      this.rows.push(row);
	      this.canChange.push(true);
      }

      this.clicked = [];
      this.hasStartOfTextfield = []
  
      // construct HTML for text fields
      for(i=0; i<this.review.length; i++) {
              this.clicked.push(false);
              this.hasStartOfTextfield.push(false);
      }
      this.redrawTable();
	    if(this.label == "entailment") {
		    instrImply = "DOES NOT IMPLY"
	    } else {
		    instrImply = "IMPLIES"
	    }
	    this.step = 0

      console.log("DONE PRESENTING");
      this.nextStep();
    },

    nextStep : function() {
           $(".err").hide();



	    console.log(this.step);
      if(this.step == 0) {
          instructions = "Read the review carefully.<br>Note that it sounds <b>POSITIVE</b>: The writer says they cannot recommend the movie enough, so they must like it.<br><br>Then click on `Next Step'."
      } else if(this.step == 1) {
	      instructions = "Can you change the review so it sounds <b>NEGATIVE</b>?<br><br>1. Click on `enough'. A text field pops up. Click on `Next Step'."
      } else if(this.step == 2) {
	      console.log(this.clicked)
	      for(i=0; i<this.clicked.length; i++) {
    	         if((this.clicked[i]) != (!(i != 6))) {
                            $(".err").show();
           	             return;
    	         }
	      }
	      instructions = "2. Enter `at all' (without quotation marks).<br><br>3. Now the text reads: `I simply can't recommend this movie at all.'<br><br>4. Now, the text sounds negative!<br><br>5. Click on `Next Step'."
      } else if(this.step == 3) {
	      //console.log(document.getElementById("input_textfield8").value);
	      //if(document.getElementById("input_textfield8").value.trim() != "sister") {
               //   $(".err").show();
		//  return;
	      //}
	      instructions = "There are other ways to make the review sound negative.<br><br>Click on `Save and try another possibility'."
              $(".nextStep").hide()
      } else if(this.step == 4) {
              $(".nextStep").show()
	      instructions = "Your task is to come up with as many ways as possible of changing the text, while leaving the words you've changed so far (`enough.') unchanged! If you click on `enough.', nothing will happen.<br><br>Click on `Next Step'."
      } else if(this.step == 5) {
	      instructions = "1. Click on `simply', `can't', and `recommend'.<br><br>2. Enter `discourage you from watching'.<br><br>3. Now the review again sounds NEGATIVE! Click on `Next Step'."
      } else if(this.step == 6) {
//	      for(i=0; i<this.clicked.length; i++) {
//    	         if((this.clicked[i]) != (!(i != 3))) {
//                            $(".err").show();
//           	             return;
//    	         }
//	      }
	      instructions = "You can add and remove a text field by clicking on a word. Click on `movie' to create a text field, and then click on it again to remove the text field.<br><br>Click on `Next Step'."
      } else if(this.step == 7) {
	      if(this.clicked[0]) {
                  $(".err").show();
		  return;
	      }
	      instructions = "There are still further ways of changing the text! Click on `Save and try another possibility'"
              $(".nextStep").hide()
      } else if(this.step == 8) {
	      instructions = "1. Click on `this' and enter `staying away from this'.<br><br>2. Now the text sounds NEGATIVE again!<br><br>Click on `Next Step'."
              $(".nextStep").show()
      } else if(this.step == 9) {
	      for(i=0; i<this.clicked.length; i++) {
    	         if((this.clicked[i]) != (!(i != 4))) {
                            $(".err").show();
           	             return;
    	         }
	      }
	      instructions = "Maybe you can come up with even more ways of changing the text, but we'll stop this example here.<br><br>Click on `Next Step'."
      } else if(this.step == 10) {
	  instructions = "For each text snippet in this HIT, you will be required to find at least one change for each text.<br>In some -- but not all -- cases, there may be additional correct changes that also satisfy the requirement. For each additional correct change that you come up with, we will bonus you 15 cents, up to a total bonus of $6.00. Click on `Next Step'."
      } else if(this.step == 11) {
	      instructions = "In this example, the review sounded POSITIVE, and your job was to change it so it sounded NEGATIVE. In this experiment, you will also see texts that sound NEGATIVE, and your job will be to change the text so it sounds POSITIVE. For each text, we will tell you which version applies; make sure to pay attention. Click on `Next Step'."
      } else if(this.step == 12) {
	      instructions = "Click on `Save and go to Next Sentence' to see another example."
      } else {
              $(".err").show();
	      return;
      }
      this.step += 1;
      $(".instr1").html(instructions);
      $(".err").hide();

    },

    tableClick : function(i) {
	    if(this.canChange[i]) {
	       this.clicked[i] = !this.clicked[i];
	       this.redrawTable();
	    }
    },

    redrawTable : function() {
  
      htmlTextFieldByRow = ["<tr>"];
      htmlTextByRow = ["<tr>"];
      for(row = 0; row < this.rows.length; row++) {
	  htmlTextFieldByRow.push("<tr>");
      	  htmlTextByRow.push("<tr>");
      }

    

      for(i=0; i<this.review.length; i++) {
	      if(this.clicked[i] & (i == 0 | !this.clicked[i-1])) {
		      lengthInColumns = 0;
		      accumulatedText = "";
		      lengthInCharacters = 0;
		      for(j=i; j<this.review.length; j++) {
			      if(!this.clicked[j]) {
				      break;
			      }
			      lengthInCharacters += this.review[j].length;
			      lengthInColumns += 1;
			      if(this.hasStartOfTextfield[j]) {
				      accumulatedText += (document.getElementById("input_textfield"+j).value + " ")
			      }
  	                      this.hasStartOfTextfield[j] = false;
		      }
                     htmlTextFieldByRow[this.rows[i]] += "<td colspan="+lengthInColumns+">" + "<textarea rows=1 cols="+lengthInCharacters+" id=input_textfield"+i+">"+accumulatedText.trim()+"</textarea>" + "</td>"
  	             this.hasStartOfTextfield[i] = true;
	      } else {
  	             this.hasStartOfTextfield[i] = false;
	             if(!this.clicked[i]) {
                          htmlTextFieldByRow[this.rows[i]] += "<td>"  + "</td>"
	             } else {
			if(i>0 & this.rows[i] != this.rows[i-1]) {
               		     for(j=i; j<this.review.length; j++) {
               		       if(!this.clicked[j]) {
               		   		break;
               		   	}
               		        htmlTextFieldByRow[this.rows[i]] += "<td>"  + "</td>"
               		    }

			}
		     }
	      }
      }
	    for(row=0; row<htmlTextFieldByRow.length; row++) {
	        htmlTextFieldByRow[row] += "</tr>"
	    }


      for(i=0; i<this.review.length; i++) {
	      sentenceHere = this.review[i]
	      if(this.clicked[i]) {
		      sentenceHere = "<strike>"+sentenceHere+"</strike>"
	      } else {
                      if(!this.canChange[i]) {
   		           sentenceHere = "<i>"+sentenceHere+"</i>"
		      } else {
   		           sentenceHere = "<b>"+sentenceHere+"</b>"
		      }

	      }
          htmlTextByRow[this.rows[i]] += '<td  onclick='+"'"+    '_s.tableClick('+i+')' +"'"+' >' + sentenceHere + "&nbsp; </td>"
      }


      html_ = ""
      for(row=0; row<htmlTextFieldByRow.length; row++) {
  	    html_ += "<table>"+htmlTextFieldByRow[row] + htmlTextByRow[row]+"</table>"
      }
      $(".review").html(html_); //stim.sentence+);



    },


    button : function() {
	    if(this.step != 4 && this.step != 8) {
                $(".err").show();
		    return;
	    }
	 allFieldsFilled = true;
	 this.filled = []

         this.responses = []
      for(i=0; i<this.review.length; i++) {
	      if(this.clicked[i] & (i == 0 | !this.clicked[i-1])) {
		      lengthInColumns = 0;
		      accumulatedText = "";
		      lengthInCharacters = 0;
		      for(j=i; j<this.review.length; j++) {
			      if(!this.clicked[j]) {
				      break;
			      }
			      lengthInColumns += 1;
			      if(this.hasStartOfTextfield[j]) {
				      accumulatedText += (" "+document.getElementById("input_textfield"+j).value)
			      }
  		      }
		      this.responses.push([i, lengthInColumns, accumulatedText]);
	      }
      }
	    console.log(this.responses);


         if (this.responses.length > 0) {
		 for(i=0; i<this.review.length; i++) {
			 if(this.clicked[i]) {
 		             this.canChange[i]=false;
				 this.clicked[i]=false;
			 }
		 }
		 this.redrawTable();
         } else {
           $(".err").show();

	 }
	    this.nextStep();
    },
	  nextSentence : function() {
		  if(this.step == 13) {
                      $(".err").hide();
   		      exp.go();
		  }
                 $(".err").show();
	  },

  });




  slides.example2 = slide({
    name : "example2",
    start : function() {
      console.log("START EXP");
      $(".err").hide();

      this.review = ["What", "an", "utter", "waste", "of", "time.", "Decent", "acting", "but", "totally", "predictable", "plot."]

      this.label = 1

      this.canChange = [];
      this.rows = [];
      aggregateLength = 0
	    row = 0;
      for(i=0; i<this.review.length; i++) {
	      aggregateLength+=this.review[i].length;
	      if(aggregateLength > 50) {
		      row += 1;
		      aggregateLength = this.review[i].length;
	      }
	      this.rows.push(row);
	      this.canChange.push(true);
      }

      this.clicked = [];
      this.hasStartOfTextfield = []
  
      // construct HTML for text fields
      for(i=0; i<this.review.length; i++) {
              this.clicked.push(false);
              this.hasStartOfTextfield.push(false);
      }
      this.redrawTable();
	    if(this.label == "entailment") {
		    instrImply = "DOES NOT IMPLY"
	    } else {
		    instrImply = "IMPLIES"
	    }
	    this.step = 0

      console.log("DONE PRESENTING");
      this.nextStep();
    },

    nextStep : function() {
           $(".err").hide();



	    console.log(this.step);
      if(this.step == 0) {
          instructions = "Read the review carefully.<br>Note that it sounds <b>NEGATIVE</b>.<br><br>Then click on `Next Step'."
      } else if(this.step == 1) {
	      instructions = "Can you change the review so it sounds <b>POSITIVE</b>?<br><br>1. Click on `an', `utter', `waste', `of', `time'. A text field pops up. Click on `Next Step'."
      } else if(this.step == 2) {
	      console.log(this.clicked)
	      for(i=0; i<this.clicked.length; i++) {
    	         if((this.clicked[i]) != (!(i != 1 && i != 2 && i != 3 && i != 4 && i != 5))) {
                            $(".err").show();
           	             return;
    	         }
	      }
	      instructions = "2. Enter `a nice surprise.'.<br><br>3. Click on `but', `totally', and enter `though somewhat'.<br><br>4. Now, the text sounds a lot more positive!<br><br>5. Click on `Next Step'."
      } else if(this.step == 3) {
	      console.log(this.clicked)
	      for(i=0; i<this.clicked.length; i++) {
    	         if((this.clicked[i]) != (!(i != 1 && i != 2 && i != 3 && i != 4 && i != 5 && i!=8 && i != 9))) {
                            $(".err").show();
           	             return;
    	         }
	      }
	      instructions = "Click on `Save and try another possibility."
              $(".nextStep").hide()
      } else if(this.step == 4) {
              $(".nextStep").show()
	      instructions = "Can you come up with another way to make it sound positive? How about changing `What' to `It first seemd like', and changing `Decent acting' to `Slow start', and changing `predictable plot' to `amazing acting.'<br><br>Now it sounds positive!<br><br>Click on `Save and go to Next Sentence' to get started with the experiment."
      } else {
              $(".err").show();
	      return;
      }
      this.step += 1;
      $(".instr1").html(instructions);
      $(".err").hide();

    },

    tableClick : function(i) {
	    if(this.canChange[i]) {
	       this.clicked[i] = !this.clicked[i];
	       this.redrawTable();
	    }
    },

    redrawTable : function() {
  
      htmlTextFieldByRow = ["<tr>"];
      htmlTextByRow = ["<tr>"];
      for(row = 0; row < this.rows.length; row++) {
	  htmlTextFieldByRow.push("<tr>");
      	  htmlTextByRow.push("<tr>");
      }

    

      for(i=0; i<this.review.length; i++) {
	      if(this.clicked[i] & (i == 0 | !this.clicked[i-1])) {
		      lengthInColumns = 0;
		      accumulatedText = "";
		      lengthInCharacters = 0;
		      for(j=i; j<this.review.length; j++) {
			      if(!this.clicked[j]) {
				      break;
			      }
			      lengthInCharacters += this.review[j].length;
			      lengthInColumns += 1;
			      if(this.hasStartOfTextfield[j]) {
				      accumulatedText += (document.getElementById("input_textfield"+j).value + " ")
			      }
  	                      this.hasStartOfTextfield[j] = false;
		      }
                     htmlTextFieldByRow[this.rows[i]] += "<td colspan="+lengthInColumns+">" + "<textarea rows=1 cols="+lengthInCharacters+" id=input_textfield"+i+">"+accumulatedText.trim()+"</textarea>" + "</td>"
  	             this.hasStartOfTextfield[i] = true;
	      } else {
  	             this.hasStartOfTextfield[i] = false;
	             if(!this.clicked[i]) {
                          htmlTextFieldByRow[this.rows[i]] += "<td>"  + "</td>"
	             } else {
			if(i>0 & this.rows[i] != this.rows[i-1]) {
               		     for(j=i; j<this.review.length; j++) {
               		       if(!this.clicked[j]) {
               		   		break;
               		   	}
               		        htmlTextFieldByRow[this.rows[i]] += "<td>"  + "</td>"
               		    }

			}
		     }
	      }
      }
	    for(row=0; row<htmlTextFieldByRow.length; row++) {
	        htmlTextFieldByRow[row] += "</tr>"
	    }


      for(i=0; i<this.review.length; i++) {
	      sentenceHere = this.review[i]
	      if(this.clicked[i]) {
		      sentenceHere = "<strike>"+sentenceHere+"</strike>"
	      } else {
                      if(!this.canChange[i]) {
   		           sentenceHere = "<i>"+sentenceHere+"</i>"
		      } else {
   		           sentenceHere = "<b>"+sentenceHere+"</b>"
		      }

	      }
          htmlTextByRow[this.rows[i]] += '<td  onclick='+"'"+    '_s.tableClick('+i+')' +"'"+' >' + sentenceHere + "&nbsp; </td>"
      }


      html_ = ""
      for(row=0; row<htmlTextFieldByRow.length; row++) {
  	    html_ += "<table>"+htmlTextFieldByRow[row] + htmlTextByRow[row]+"</table>"
      }
      $(".review").html(html_); //stim.sentence+);



    },


    button : function() {
	    if(this.step != 4) {
                $(".err").show();
		    return;
	    }
	 allFieldsFilled = true;
	 this.filled = []

         this.responses = []
      for(i=0; i<this.review.length; i++) {
	      if(this.clicked[i] & (i == 0 | !this.clicked[i-1])) {
		      lengthInColumns = 0;
		      accumulatedText = "";
		      lengthInCharacters = 0;
		      for(j=i; j<this.review.length; j++) {
			      if(!this.clicked[j]) {
				      break;
			      }
			      lengthInColumns += 1;
			      if(this.hasStartOfTextfield[j]) {
				      accumulatedText += (" "+document.getElementById("input_textfield"+j).value)
			      }
  		      }
		      this.responses.push([i, lengthInColumns, accumulatedText]);
	      }
      }
	    console.log(this.responses);


         if (this.responses.length > 0) {
		 for(i=0; i<this.review.length; i++) {
			 if(this.clicked[i]) {
 		             this.canChange[i]=false;
				 this.clicked[i]=false;
			 }
		 }
		 this.redrawTable();
         } else {
           $(".err").show();

	 }
	    this.nextStep();
    },
	  nextSentence : function() {
		  if(this.step == 5) {
                      $(".err").hide();
   		      exp.go();
		  }
                 $(".err").show();
	  },

  });









  slides.multi_slider_context = slide({
    name : "multi_slider_context",
    present : stimuliContext,
    present_handle : function(stim) {
      console.log("START EXP");
      console.log(stim);
      $(".err").hide();
      this.stim = stim; //FRED: allows you to access stim in helpers
      console.log(stim.sentence);
      stim.sentence = stim.sentence.split(" ");
      sentence = stim.sentence;
      console.log(sentence);

	    this.allResponses = [];
	    this.canChange = [];
      this.rows = [];
      aggregateLength = 0
	    row = 0;
      for(i=0; i<this.stim.sentence.length; i++) {
	      aggregateLength+=this.stim.sentence[i].length;
	      if(aggregateLength > 60) {
		      row += 1;
		      aggregateLength = this.stim.sentence[i].length;
	      }
	      this.rows.push(row);
	      this.canChange.push(true);
      }

      this.clicked = [];
      this.hasStartOfTextfield = []
  
      // construct HTML for text fields
      for(i=0; i<this.stim.sentence.length; i++) {
              this.clicked.push(false);
              this.hasStartOfTextfield.push(false);
      }
      this.redrawTable();
	    if(this.stim.label == 1) {
		    instrImply = "NEGATIVE"
		    instrImply1 = "POSITIVE"
	    } else {
		    instrImply = "POSITIVE"
		    instrImply1 = "NEGATIVE"
	    }
      $(".instr1").html("The review sounds "+instrImply1+".<br>Can you change the text so it sounds "+instrImply+"?<br>");

      console.log("DONE PRESENTING");
$(".bonus").html("$"+Math.min(6.00,Math.round(exp.bonus*100)/100));
      this.stim = stim;
    },



    tableClick : function(i) {
	    if(this.canChange[i]) {
	       this.clicked[i] = !this.clicked[i];
	       this.redrawTable();
	    }
    },

    redrawTable : function() {
            $(".err").hide();
 
      htmlTextFieldByRow = ["<tr>"];
      htmlTextByRow = ["<tr>"];
      for(row = 0; row < this.rows.length; row++) {
	  htmlTextFieldByRow.push("<tr>");
      	  htmlTextByRow.push("<tr>");
      }

    

      for(i=0; i<this.stim.sentence.length; i++) {
	      if(this.clicked[i] & (i == 0 | !this.clicked[i-1])) {
		      lengthInColumns = 0;
		      accumulatedText = "";
		      lengthInCharacters = 0;
		      for(j=i; j<this.stim.sentence.length; j++) {
			      if(!this.clicked[j]) {
				      break;
			      }
			      lengthInCharacters += this.stim.sentence[j].length;
			      lengthInColumns += 1;
			      if(this.hasStartOfTextfield[j]) {
				      accumulatedText += (document.getElementById("input_textfield"+j).value + " ")
			      }
  	                      this.hasStartOfTextfield[j] = false;
		      }
                     htmlTextFieldByRow[this.rows[i]] += "<td colspan="+lengthInColumns+">" + "<textarea rows=1 cols="+lengthInCharacters+" id=input_textfield"+i+">"+accumulatedText.trim()+"</textarea>" + "</td>"
  	             this.hasStartOfTextfield[i] = true;
	      } else {
  	             this.hasStartOfTextfield[i] = false;
	             if(!this.clicked[i]) {
                          htmlTextFieldByRow[this.rows[i]] += "<td>"  + "</td>"
	             } else {
			if(i>0 & this.rows[i] != this.rows[i-1]) {
               		     for(j=i; j<this.stim.sentence.length; j++) {
               		       if(!this.clicked[j]) {
               		   		break;
               		   	}
               		        htmlTextFieldByRow[this.rows[i]] += "<td>"  + "</td>"
               		    }

			}
		     }
	      }
      }
	    for(row=0; row<htmlTextFieldByRow.length; row++) {
	        htmlTextFieldByRow[row] += "</tr>"
	    }


      for(i=0; i<this.stim.sentence.length; i++) {
	      sentenceHere = this.stim.sentence[i]
	      if(this.clicked[i]) {
		      sentenceHere = "<strike>"+sentenceHere+"</strike>"
	      } else {
                      if(!this.canChange[i]) {
   		           sentenceHere = "<i>"+sentenceHere+"</i>"
		      } else {
   		           sentenceHere = "<b>"+sentenceHere+"</b>"
		      }
	      }
          htmlTextByRow[this.rows[i]] += '<td  onclick='+"'"+    '_s.tableClick('+i+')' +"'"+' >' + sentenceHere + "&nbsp; </td>"
      }


      html_ = ""
      for(row=0; row<htmlTextFieldByRow.length; row++) {
  	    html_ += "<table>"+htmlTextFieldByRow[row] + htmlTextByRow[row]+"</table>"
      }
      $(".review").html(html_); //stim.sentence+);
    },


    getResponses : function() {
         this.responses = []
         for(i=0; i<this.stim.sentence.length; i++) {
	      if(this.clicked[i] & (i == 0 | !this.clicked[i-1])) {
		      lengthInColumns = 0;
		      accumulatedText = "";
		      lengthInCharacters = 0;
		      for(j=i; j<this.stim.sentence.length; j++) {
			      if(!this.clicked[j]) {
				      break;
			      }
			      lengthInColumns += 1;
			      if(this.hasStartOfTextfield[j]) {
				      accumulatedText += (" "+document.getElementById("input_textfield"+j).value)
			      }
  		      }
		      this.responses.push([i, lengthInColumns, accumulatedText]);
	      }
        }
        console.log(this.responses);
	    if(this.responses.length > 0) {
               this.allResponses.push(this.responses);
	    }
	    return this.responses.length
    },

    button : function() {
	 allFieldsFilled = true;
	 this.filled = []
         

         if (this.getResponses() > 0) {
		 for(i=0; i<this.stim.sentence.length; i++) {
			 if(this.clicked[i]) {
 		             this.canChange[i]=false;
				 this.clicked[i]=false;
			 }
		 }
		 this.redrawTable();
         } else {
           $(".err").show();

	 }
    },
	  nextSentence : function() {
		  this.getResponses();
		  if(this.allResponses.length == 0) {
           $(".err").show();
			  return;

		  }
           $(".err").hide();

           this.log_responses();
           _stream.apply(this); //use exp.go() if and only if there is no "present" data.
	  },
    log_responses : function() {
        //console.log(this.stim.condition);
	console.log(this.allResponses);
	    exp.bonus += 0.15 * (this.allResponses.length-1)
        dataForThisTrial = ({
          "completion" : this.allResponses,
          "sentence" : this.stim.sentence,
          "original" : this.stim.original,
          "sensitivity" : this.stim.sensitivity,
          "label" : this.stim.label,
          "slide_number" : exp.phase
        });
        exp.data_trials.push(dataForThisTrial);
	    console.log("SENDING TO SERVER");
	    console.log(exp.data_trials[exp.data_trials.length-1]);

      dataExperiment= {
          "time_in_minutes" : (Date.now() - exp.startT)/60000,
	  "ProlificURL" : window.location.href
      };



xhr = new XMLHttpRequest();

serverIndex = _.sample([1,2,3,4,5,6,7,8,9,10], 1)


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
	      bonus : "$"+Math.min(6.00,Math.round(exp.bonus*100)/100),
        education : $("#education").val(),
//        colorblind : $("#colorblind").val(),
        comments : $("#comments").val(),
        suggested_pay : $("#suggested_pay").val(),
        condition : exp.condition,
        context_first : exp.context_first,
        tutorial : exp.tutorial,
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



	    // set `Content-Type` header
	     xhr.setRequestHeader('Content-Type', 'application/json');
	    //
	    // // send rquest with JSON payload
	     xhr.send(JSON.stringify(exp.data));
      $(".redirect_prolific").html("Please click on this link to record your participation: <br><br><b><a href='https://app.prolific.co/submissions/complete?cc=4A78476B'>Record Participation</a></b><br><br>If you do not do this, you will NOT GET PAID.");

    }
  });

  return slides;
}

/// init ///
function init() {
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
exp.structure.push( 'example1')
exp.structure.push( 'example2')
//exp.structure.push( 'example3')
//exp.structure.push( 'example4')
//exp.structure.push( 'example5')
////exp.structure.push( 'example6')
   exp.structure.push( 'multi_slider_context')
//   exp.structure.push( 'motivation')
//   exp.structure.push( 'multi_slider_context2')

exp.structure.push( 'subj_info')
exp.structure.push( 'thanks');

exp.bonus = 0;

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
