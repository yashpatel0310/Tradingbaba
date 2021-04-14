from django import forms


stock= [
    ('AAPL', 'AAPL'),
    ('MSFT', 'MSFT'),
    ('TSLA', 'TSLA'),
    ('GOOGL', 'GOOGL'),
    ]

class stocklist(forms.Form):
    stock_tick= forms.CharField(label='', widget=forms.Select(choices=stock,attrs={'style': 'width: 200px; margin-top:20px; height: 40px; font-size: 25px; font-family:Copperplate Gothic; border-radius:5px; background-color: #27293D; box-shadow: 0px 8px 24px 1px #00000080; float:right; margin-right: 160px '}))
