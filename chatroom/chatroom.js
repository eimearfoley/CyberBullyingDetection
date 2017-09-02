(function() {

    var sentmessage;
    var message;


    document.addEventListener('DOMContentLoaded', init, false);

    function init() {
        click = document.querySelector('#click');
        message = document.querySelector('#message');
        message.addEventListener('keypress', set_link, false);
        click.addEventListener('click', change, false);
    }
        
    function set_link() {
        click.innerHTML = '<button class="button">Send</button>';
    }


    function change() {
        var url = 'getmessage.py';
        request = new XMLHttpRequest();
        request.addEventListener('readystatechange', handle_response, false);
        request.open('GET', url, true);
        console.log(request.responseText);
        request.send(null);
    }

    function handle_response() {
        if ( request.readyState === 4 ) {
            if ( request.status === 200 ) {
                console.log(request.responseText);
                if ( request.responseText.trim() === 'error' ) {
                    display.innerHTML = 'There has been an error';
                } else {
                    display.innerHTML = request.responseText;
                }
            }
        }
    }

})();
