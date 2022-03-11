import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from prediction.predictor import Predictor
from utils.data_tools import remove_special_symbols, tokenize_and_lemmatize_and_steam_text


def _clean_text(text):
    clean_text = remove_special_symbols(text=text)
    tokens, lemmas = tokenize_and_lemmatize_and_steam_text(clean_text)
    return ' '.join(lemmas)




if __name__ == "__main__":
    print("STARTING PREDICTION")

    predictor = Predictor()

    raw_text_list = [
        ("Hi Jane, Respond with YES to confirm your appointment for tomorrow at 10 am or NO to reschedule.",0),
        ("If you'd like a health challenge, text back TRY",0),
        ("We apologize for that! We are firing up a fresh batch of fries for you now and will bring them out to your car as soon as you get here. Just text back 'HERE' when you arrive. - Sarah, Manager",0),
        ("How satisfied were you with your recent dentist appointment? [1 Very Unsatisfied] [5 Very Satisfied]",0),
        ("To confirm your $10 donation to the American Red Cross Disaster Relief reply with YES. Reply with HELP for help or visit redcross.org/m",0),
        ("Hi Matt, your payment is one week past due. Please use the link below to make your payment. Thank you. bit.ly/inv12",0),
        ("Redbox: Psst. Follow us on Instagram, because we just posted a code for a free movie night: https://www.instagram.com/redbox/",0),
        ("Hi Laura. This is Jon at the Audi Dealership. Your vehicle has been checked in. You can view the status here: https://mds.im/r/365",0),
        ("Hi! I would like to place an order.",0),
        ("Where can I order Pizza?",0),
        ("Which slot machine will make me win?",0),
        ("Can you get me tickets to the movies?",0),
        ("BANK OF AMERICA EDD: Your Bank of America EDD Prepaid Debit Card has been temporarily suspended and requires verification to activate. Please click the link below to re-activate your Bank of America EDD Prepaid Card to continue using. https://visarprocessingeddcard.com/reactive",1),
        ("FRM:ChaseBank-Mobile-APP-ID-n07jn SUBJ:Contact: 8125652928 #Now MSG:ID: 3386 CardLocked!",1),
        ("Costco: The code 61543 printed on your receipt from 18 came in 1st in our Earpods draw: j2hbv.info/lYWlcZEmfG",1),
        ("Your FREE PS5 is Here! Best Buy PS5 Contest! Thanks for shopping with us. BestBuy is giving away 10 PS5's! Click to see if you are one of the 10 lucky winners! conitnue832.com/iZu0Ww5",1),
        ("$130 freebies from Costco, that's our Covid-19 stimulus package for all loyal Costco members! a6er.info/ fill in customer survey in return please",1),
        ("Your $1,820 is now available from Walmart COVID-19 Relief Program, Apply here: https://bit.ly/WalmartPandemicsReliefs",1),
        ("Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123456 to claim now.",1),
        ("Your IRS tax refund is pending acceptance. Must accept within 24 hours: http://bit.ly/sdfsdf",1),
        ("Amazon is sending you a refund of $32,64. Please reply with your bank account and routing number to receive your refund.",1),
        ("Wells Fargo Bank: Your account is temporarily locked. Please log in at http://gool.gl/2a234 to secure your account.",1),
        ("Hello, your FEDEX package with tracking code DZ-8342-FY34 is waiting for you to set delivery preferences: c4lrs.info/Gm08s43vz1",1),
        ("Apple Notification. Your Apple iCloud ID expires today. Log in to prevent deletion http://apple.id/user-auth/online",1),
        ("((Coinbase)) Amount received 2.221 Bitcoin BTC ($18,421 USD,), Please confirm transaction: http://bit.do/CoinBase432194-543242",1),
        ("URGENT: Your grandson was arrested last night in Mexico. Need bail money immediately Western Union Wire $9,500 http://goo.gl/ndf4g5",1),
        ("The current leading bid is 151. To pause this auction send OUT. Customer Care: 08718726270", 1)
    ]

    clean_text_list = []

    for pair in raw_text_list:
        raw_text = pair[0]
        text = _clean_text(raw_text)
        clean_text_list.append((text, pair[1]))

    hit = 0
    miss = 0

    for index,pair in enumerate(clean_text_list):
        text = pair[0]
        result = predictor.predict(text)
        expected_result = "spam" if pair[1] == 1 else "ham"
        ham_or_spam = "spam" if result[0] == 1 else "ham"
        if ham_or_spam == expected_result:
            hit += 1 
        else:
            miss += 1
        print(f'PREDICTING TEXT NUMBER {index} \n PREDICTED CLASS: {ham_or_spam} \n EXPECTED CLASS: {expected_result}')
    print()
    print(f'Acertos: {hit} ({round((hit/(hit+miss)) * 100)}%)')
    print(f'Erros {miss} ({round((miss/(hit+miss)) * 100)}%)')
